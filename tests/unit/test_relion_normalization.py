"""RELION norm/scale formulas, independent of the refinement controller."""

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.types import NoiseStats
from recovar.em.dense_single_volume.relion_normalization import update_relion_norm_scale_corrections

pytestmark = pytest.mark.unit


class TestRelionNormalization:
    def test_relion_norm_scale_update_matches_relion_formula(self):
        """Native updater reconstructs RELION's normcorr/group-scale state."""
        stats = NoiseStats(
            wsum_sigma2_noise=jnp.array([0.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([0.0], dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=4.0,
            wsum_norm_correction=jnp.array([2.0, 8.0, 18.0, 0.5], dtype=jnp.float32),
            wsum_scale_correction_xa=jnp.array([1.0, 100.0, 1.0], dtype=jnp.float32),
            wsum_scale_correction_aa=jnp.array([1.0, 10.0, 0.0], dtype=jnp.float32),
        )
        group_ids = np.array([0, 1, 1, 2], dtype=np.int64)
        old_group_scale = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        old_scale = old_group_scale[group_ids]
        old_image_corr = np.array([2.0, 1.0, 4.0, 4.0], dtype=np.float64)

        got = update_relion_norm_scale_corrections(
            noise_stats_per_half=[stats, stats],
            image_corrections_per_half=[old_image_corr, old_image_corr],
            scale_corrections_per_half=[old_scale, old_scale],
            group_ids_per_half=[group_ids, group_ids],
            group_scale_corrections_per_half=[old_group_scale, old_group_scale],
        )

        expected_normcorr = np.array([1.0, 8.0, 3.0, 1.0], dtype=np.float64)
        expected_avg_norm = float(np.mean(expected_normcorr))
        scale_target = np.array([1.0, 10.0, 1.0], dtype=np.float64)
        clipped = np.array([1.0, 5.0, 1.0], dtype=np.float64)
        expected_group_scale = clipped / ((1.0 * clipped[0] + 2.0 * clipped[1] + clipped[2]) / 4.0)
        expected_scale = expected_group_scale[group_ids]
        expected_image_corr = (expected_avg_norm / expected_normcorr) * expected_scale

        np.testing.assert_allclose(np.asarray(got.norm_corrections_per_half[0]), expected_normcorr, rtol=1e-6)
        assert got.avg_norm_correction_per_half[0] == pytest.approx(expected_avg_norm)
        np.testing.assert_allclose(np.asarray(got.group_scale_corrections_per_half[0]), expected_group_scale, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(got.scale_corrections_per_half[0]), expected_scale, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(got.image_corrections_per_half[0]), expected_image_corr, rtol=1e-6)

    def test_relion_norm_update_divides_by_retained_posterior_mass(self):
        """RELION divides its unweighted normcorr sum by significant support mass."""
        stats = NoiseStats(
            wsum_sigma2_noise=jnp.array([0.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([0.0], dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=3.5,
            wsum_norm_correction=jnp.array([2.0, 8.0, 18.0, 0.5], dtype=jnp.float32),
        )
        group_ids = np.zeros(4, dtype=np.int64)
        old_scale = np.ones(4, dtype=np.float64)
        old_image_corr = np.array([2.0, 0.5, 2.0, 1.0], dtype=np.float64)

        got = update_relion_norm_scale_corrections(
            noise_stats_per_half=[stats, stats],
            image_corrections_per_half=[old_image_corr, old_image_corr],
            scale_corrections_per_half=[old_scale, old_scale],
            group_ids_per_half=[group_ids, group_ids],
            do_scale_correction=False,
        )

        expected_normcorr = np.array([1.0, 8.0, 3.0, 1.0], dtype=np.float64)
        expected_avg_norm = float(np.sum(expected_normcorr) / stats.sumw)
        expected_image_corr = expected_avg_norm / expected_normcorr
        np.testing.assert_allclose(np.asarray(got.norm_corrections_per_half[0]), expected_normcorr, rtol=1e-6)
        assert got.avg_norm_correction_per_half[0] == pytest.approx(expected_avg_norm)
        np.testing.assert_allclose(np.asarray(got.image_corrections_per_half[0]), expected_image_corr, rtol=1e-6)

    def test_relion_norm_scale_update_accepts_single_active_class3d_half(self):
        """Class3D-style single-half runs still update active-half corrections."""
        active = NoiseStats(
            wsum_sigma2_noise=jnp.array([0.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([0.0], dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=2.0,
            wsum_norm_correction=jnp.array([2.0, 8.0], dtype=jnp.float32),
            wsum_scale_correction_xa=jnp.array([2.0], dtype=jnp.float32),
            wsum_scale_correction_aa=jnp.array([1.0], dtype=jnp.float32),
        )
        empty = NoiseStats(
            wsum_sigma2_noise=jnp.array([0.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([0.0], dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=0.0,
        )

        got = update_relion_norm_scale_corrections(
            noise_stats_per_half=[active, empty],
            group_ids_per_half=[np.zeros(2, dtype=np.int64), np.zeros(0, dtype=np.int64)],
        )

        np.testing.assert_allclose(np.asarray(got.norm_corrections_per_half[0]), [2.0, 4.0], rtol=1e-6)
        assert got.avg_norm_correction_per_half[0] == pytest.approx(3.0)
        np.testing.assert_allclose(np.asarray(got.group_scale_corrections_per_half[0]), [1.0], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(got.image_corrections_per_half[0]), [1.5, 0.75], rtol=1e-6)
        assert np.asarray(got.image_corrections_per_half[1]).shape == (0,)
        assert np.asarray(got.scale_corrections_per_half[1]).shape == (0,)

    def test_relion_norm_scale_update_preserves_explicit_absent_groups(self):
        """Half-local missing groups retain RELION's full model-group axis."""
        stats = NoiseStats(
            wsum_sigma2_noise=jnp.array([0.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([0.0], dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=2.0,
            wsum_norm_correction=jnp.ones(2, dtype=jnp.float32),
            wsum_scale_correction_xa=jnp.array([2.0, 0.0, 4.0, 0.0, 0.0], dtype=jnp.float32),
            wsum_scale_correction_aa=jnp.array([1.0, 0.0, 2.0, 0.0, 0.0], dtype=jnp.float32),
        )
        group_ids = np.asarray([0, 2], dtype=np.int64)

        got = update_relion_norm_scale_corrections(
            noise_stats_per_half=[stats, stats],
            group_ids_per_half=[group_ids, group_ids],
            group_count_per_half=[5, 5],
            do_norm_correction=False,
        )

        assert np.asarray(got.group_scale_corrections_per_half[0]).shape == (5,)
        assert np.asarray(got.scale_corrections_per_half[0]).shape == (2,)
        np.testing.assert_allclose(np.asarray(got.group_scale_corrections_per_half[0])[[1, 3, 4]], 0.5)

    def test_relion_norm_scale_update_skips_firstiter_cc_scale_only(self):
        """RELION firstiter-CC still updates normcorr but keeps old scales."""
        stats = NoiseStats(
            wsum_sigma2_noise=jnp.array([0.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([0.0], dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=2.0,
            wsum_norm_correction=jnp.array([2.0, 8.0], dtype=jnp.float32),
            wsum_scale_correction_xa=jnp.array([100.0, 1.0], dtype=jnp.float32),
            wsum_scale_correction_aa=jnp.array([10.0, 1.0], dtype=jnp.float32),
        )
        group_ids = np.array([0, 1], dtype=np.int64)
        old_group_scale = np.array([1.0, 2.0], dtype=np.float64)
        old_scale = old_group_scale[group_ids]
        old_image_corr = np.array([1.0, 1.0], dtype=np.float64)

        got = update_relion_norm_scale_corrections(
            noise_stats_per_half=[stats, stats],
            image_corrections_per_half=[old_image_corr, old_image_corr],
            scale_corrections_per_half=[old_scale, old_scale],
            group_ids_per_half=[group_ids, group_ids],
            group_scale_corrections_per_half=[old_group_scale, old_group_scale],
            relion_firstiter_cc_this_iter=True,
        )

        expected_normcorr = np.array([2.0, 8.0], dtype=np.float64)
        expected_avg_norm = 5.0
        expected_image_corr = (expected_avg_norm / expected_normcorr) * old_scale
        np.testing.assert_allclose(np.asarray(got.group_scale_corrections_per_half[0]), old_group_scale)
        np.testing.assert_allclose(np.asarray(got.scale_corrections_per_half[0]), old_scale)
        np.testing.assert_allclose(np.asarray(got.image_corrections_per_half[0]), expected_image_corr, rtol=1e-6)

    def test_relion_norm_scale_update_changes_two_native_groups_differently(self):
        stats = NoiseStats(
            wsum_sigma2_noise=jnp.array([0.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([0.0], dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=4.0,
            wsum_norm_correction=jnp.ones(4, dtype=jnp.float32),
            wsum_scale_correction_xa=jnp.array([2.0, 6.0], dtype=jnp.float32),
            wsum_scale_correction_aa=jnp.array([1.0, 2.0], dtype=jnp.float32),
        )
        group_ids = np.asarray([0, 1, 0, 1], dtype=np.int64)
        old_scale = np.ones(4, dtype=np.float64)

        got = update_relion_norm_scale_corrections(
            noise_stats_per_half=[stats, stats],
            scale_corrections_per_half=[old_scale, old_scale],
            group_ids_per_half=[group_ids, group_ids],
            do_norm_correction=False,
        )

        group_scale = np.asarray(got.group_scale_corrections_per_half[0], dtype=np.float64)
        assert group_scale.shape == (2,)
        assert group_scale[0] != pytest.approx(group_scale[1])
        np.testing.assert_allclose(
            np.asarray(got.scale_corrections_per_half[0], dtype=np.float64),
            group_scale[group_ids],
            rtol=1e-6,
        )

    def test_relion_norm_scale_update_preserves_zero_norm_residual_rows(self):
        """Images with no posterior norm mass keep their previous finite correction."""
        stats = NoiseStats(
            wsum_sigma2_noise=jnp.array([0.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([0.0], dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=2.0,
            wsum_norm_correction=jnp.array([12.5, 0.0, 12.5], dtype=jnp.float32),
        )
        group_ids = np.zeros(3, dtype=np.int64)
        old_scale = np.ones(3, dtype=np.float64)
        old_image_corr = np.array([1.0, 2.0, 1.0], dtype=np.float64)

        got = update_relion_norm_scale_corrections(
            noise_stats_per_half=[stats, stats],
            image_corrections_per_half=[old_image_corr, old_image_corr],
            scale_corrections_per_half=[old_scale, old_scale],
            group_ids_per_half=[group_ids, group_ids],
            avg_norm_correction_per_half=[5.0, 5.0],
            do_scale_correction=False,
        )

        np.testing.assert_allclose(np.asarray(got.image_corrections_per_half[0]), old_image_corr, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(got.norm_corrections_per_half[0]), [5.0, 2.5, 5.0], rtol=1e-6)
        assert got.avg_norm_correction_per_half[0] == pytest.approx(5.0)
        assert got.zero_norm_residual_counts == [1, 1]
        assert np.all(np.isfinite(np.asarray(got.image_corrections_per_half[0])))
