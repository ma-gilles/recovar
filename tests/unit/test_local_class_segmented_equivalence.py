"""One segmented pass must reproduce the per-class calls it replaces.

The K-class local route runs a probe pass and an M-step pass per class. With the
classes laid out as segments of one bucket's row axis the engine scores the joint
class-by-pose posterior once. These tests hold that one pass to what the per-class
calls produce on the same inputs: the same reconstruction accumulators per class,
the same class evidence and responsibilities, and the same discrete winners.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from helpers.em_arrays import _hermitian_volume, _make_rotations
from recovar.em.classification.k_class import run_local_k_class_em
from recovar.em.local.local_layout import LocalHypothesisLayout
from test_refine_relion_mode import IMAGE_SIZE, VOLUME_SHAPE, MockDataset

pytestmark = pytest.mark.unit


def _layout(n_images, counts, *, seed, n_trans=1, n_global=4):
    counts = np.asarray(counts, dtype=np.int32)
    total = int(counts.sum())
    rotations = np.asarray(_make_rotations(total, seed=seed), dtype=np.float32)
    return LocalHypothesisLayout(
        n_global_rotations=n_global,
        n_pixels=n_global,
        n_psi=1,
        rotation_offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        rotation_ids_flat=np.concatenate([np.arange(c) % n_global for c in counts]).astype(np.int32),
        rotations_flat=rotations,
        rotation_log_priors_flat=np.zeros(total, dtype=np.float32),
        rotation_counts=counts,
        translation_grid=np.zeros((n_trans, 2), dtype=np.float32),
        translation_log_priors=np.zeros((n_images, n_trans), dtype=np.float32),
        rotation_posterior_ids_flat=np.concatenate([np.arange(c) % n_global for c in counts]).astype(np.int32),
        sample_mask_flat=np.ones((total, n_trans), dtype=bool),
    )


def _run(dataset, means, noise_variance, layouts, *, segmented, priors, accumulate_noise=False):
    return run_local_k_class_em(
        dataset,
        means,
        noise_variance,
        layouts,
        "linear_interp",
        class_log_priors=priors,
        image_batch_size=2,
        rotation_block_size=4,
        current_size=None,
        reconstruct_significant_only=False,
        accumulate_noise=accumulate_noise,
        return_best_pose_details=True,
        segmented_class_rows=segmented,
    )


@pytest.mark.parametrize(
    "counts_by_class",
    [
        ([2, 2], [2, 2]),          # equal rows: the segment width fits both classes exactly
        ([3, 1], [1, 3]),          # unequal rows: each class is padded to the image's widest
    ],
    ids=["equal-rows", "unequal-rows"],
)
def test_segmented_pass_matches_per_class_calls(counts_by_class):
    rng = np.random.default_rng(11)
    n_images = 2
    dataset = MockDataset(n_images, rng)
    means = jnp.stack([_hermitian_volume(VOLUME_SHAPE, seed=211), _hermitian_volume(VOLUME_SHAPE, seed=307)], axis=0)
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    layouts = [
        _layout(n_images, counts_by_class[0], seed=17),
        _layout(n_images, counts_by_class[1], seed=29),
    ]
    priors = np.log(np.asarray([0.4, 0.6], dtype=np.float64))

    baseline = _run(dataset, means, noise_variance, layouts, segmented=False, priors=priors)
    segmented = _run(dataset, means, noise_variance, layouts, segmented=True, priors=priors)

    for class_index in range(2):
        np.testing.assert_allclose(
            np.asarray(segmented.Ft_y[class_index]), np.asarray(baseline.Ft_y[class_index]),
            rtol=5e-3, atol=1e-5, err_msg=f"Ft_y class {class_index}",
        )
        np.testing.assert_allclose(
            np.asarray(segmented.Ft_ctf[class_index]), np.asarray(baseline.Ft_ctf[class_index]),
            rtol=5e-3, atol=1e-5, err_msg=f"Ft_ctf class {class_index}",
        )
    np.testing.assert_allclose(
        np.asarray(segmented.class_responsibilities), np.asarray(baseline.class_responsibilities),
        rtol=5e-3, atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(segmented.class_posterior_sums), np.asarray(baseline.class_posterior_sums),
        rtol=5e-3, atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(segmented.stats.log_evidence_per_image),
        np.asarray(baseline.stats.log_evidence_per_image),
        rtol=1e-5, atol=1e-5,
    )
    for class_index in range(2):
        np.testing.assert_allclose(
            np.asarray(segmented.per_class_stats[class_index].log_evidence_per_image),
            np.asarray(baseline.per_class_stats[class_index].log_evidence_per_image),
            rtol=1e-5, atol=1e-5, err_msg=f"class {class_index} log evidence",
        )
        np.testing.assert_allclose(
            np.asarray(segmented.per_class_stats[class_index].rotation_posterior_sums),
            np.asarray(baseline.per_class_stats[class_index].rotation_posterior_sums),
            rtol=5e-3, atol=1e-5, err_msg=f"class {class_index} angular posterior",
        )
    # Discrete decisions agree exactly.
    np.testing.assert_array_equal(
        np.asarray(segmented.per_class_hard_assignments), np.asarray(baseline.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(np.asarray(segmented.class_assignments), np.asarray(baseline.class_assignments))
    np.testing.assert_array_equal(np.asarray(segmented.pose_assignments), np.asarray(baseline.pose_assignments))


def test_segmented_pass_matches_per_class_noise_in_aggregate():
    """RELION sums noise over classes, so the joint pass's single accumulation is the sum."""
    rng = np.random.default_rng(13)
    n_images = 2
    dataset = MockDataset(n_images, rng)
    means = jnp.stack([_hermitian_volume(VOLUME_SHAPE, seed=401), _hermitian_volume(VOLUME_SHAPE, seed=409)], axis=0)
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    layouts = [_layout(n_images, [2, 2], seed=31), _layout(n_images, [2, 2], seed=37)]
    priors = np.log(np.asarray([0.5, 0.5], dtype=np.float64))

    baseline = _run(dataset, means, noise_variance, layouts, segmented=False, priors=priors, accumulate_noise=True)
    segmented = _run(dataset, means, noise_variance, layouts, segmented=True, priors=priors, accumulate_noise=True)

    assert baseline.aggregate_noise_stats is not None and segmented.aggregate_noise_stats is not None
    for field in ("wsum_sigma2_noise", "wsum_img_power"):
        np.testing.assert_allclose(
            np.asarray(getattr(segmented.aggregate_noise_stats, field), dtype=np.float64),
            np.asarray(getattr(baseline.aggregate_noise_stats, field), dtype=np.float64),
            rtol=5e-3, atol=1e-5, err_msg=field,
        )
    np.testing.assert_allclose(
        float(segmented.aggregate_noise_stats.sumw), float(baseline.aggregate_noise_stats.sumw),
        rtol=5e-3, atol=1e-5,
    )


def test_segmented_rows_refuse_external_normalization():
    rng = np.random.default_rng(17)
    dataset = MockDataset(2, rng)
    means = jnp.stack([_hermitian_volume(VOLUME_SHAPE, seed=503)] * 2, axis=0)
    layouts = [_layout(2, [1, 1], seed=41), _layout(2, [1, 1], seed=43)]
    with pytest.raises(NotImplementedError, match="external normalization_log_evidence"):
        run_local_k_class_em(
            dataset, means, jnp.ones(IMAGE_SIZE, dtype=jnp.float32), layouts, "linear_interp",
            class_log_priors=np.zeros(2), image_batch_size=2, rotation_block_size=4, current_size=None,
            normalization_log_evidence=np.zeros(2), segmented_class_rows=True,
        )
