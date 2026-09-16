"""One segmented pass must reproduce the per-class calls it replaces.

The K-class local route runs a probe pass and an M-step pass per class. With the
classes laid out as segments of one bucket's row axis the engine scores the joint
class-by-pose posterior once. These tests hold that one pass to what the per-class
calls produce on the same inputs.

The fixture has to keep every class materially populated, or the comparison is
vacuous: with a peaked posterior one class takes all the mass and its rivals'
accumulators and angular sums are exactly zero, so comparing them proves nothing.
An earlier version of these tests had exactly that defect. The noise level below is
chosen so both classes keep real responsibility (minimum about 0.08 per image, class
masses about 2.5 and 3.5 of 6 images), the rows per class are unequal so the segment
padding is exercised, the translations are nontrivial, and every row carries
canonical source Euler metadata.

Tolerances come from the measured residuals rather than convention: they sit at the
reduction-order floor of each precision, about 1e-7 relative in float32 and about
3e-16 in float64, so the comparison would fail on a real behavior change.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers.em_arrays import _hermitian_volume, _make_rotations
from recovar.em.classification.k_class import run_local_k_class_em
from recovar.em.local import local_big_jit
from recovar.em.local.local_layout import LocalHypothesisLayout
from test_refine_relion_mode import IMAGE_SIZE, VOLUME_SHAPE, MockDataset

pytestmark = pytest.mark.unit

N_IMAGES = 6
# Both classes keep real posterior mass at this noise level; see the module docstring.
NOISE_VARIANCE = 1.0e3
TRANSLATIONS = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
CANONICAL_EULERS = np.asarray([360.0, 0.0, 0.0], dtype=np.float64)
# Enough candidate rows that the significant-only reconstruction route is actually
# selected: it needs local support of at least n_images / 0.25 rows. With the earlier
# 12 rows the engine took the dense in-bucket adjoint instead, so the tests never
# reached the route that had the class-routing bug.
ROWS_BY_CLASS = ([7, 5, 6, 6, 5, 7], [5, 7, 6, 6, 7, 5])


def _layout(counts, *, seed, n_global=4):
    counts = np.asarray(counts, dtype=np.int32)
    total = int(counts.sum())
    n_trans = TRANSLATIONS.shape[0]
    ids = np.concatenate([np.arange(count) % n_global for count in counts]).astype(np.int32)
    # RELION's canonical source angles for these rows; the engine must publish these
    # values, not angles recovered from the rotation matrices.
    source_eulers = np.tile(CANONICAL_EULERS, (total, 1))
    source_eulers[:, 2] = np.arange(total, dtype=np.float64)
    return LocalHypothesisLayout(
        n_global_rotations=n_global,
        n_pixels=n_global,
        n_psi=1,
        rotation_offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        rotation_ids_flat=ids,
        rotations_flat=np.asarray(_make_rotations(total, seed=seed), dtype=np.float32),
        rotation_log_priors_flat=np.zeros(total, dtype=np.float32),
        rotation_counts=counts,
        translation_grid=TRANSLATIONS,
        translation_log_priors=np.zeros((N_IMAGES, n_trans), dtype=np.float32),
        rotation_posterior_ids_flat=ids,
        sample_mask_flat=np.ones((total, n_trans), dtype=bool),
        source_eulers_flat=source_eulers,
    )


def _fixture(float64: bool):
    base = np.asarray(_hermitian_volume(VOLUME_SHAPE, seed=211))
    perturbation = np.asarray(_hermitian_volume(VOLUME_SHAPE, seed=307))
    means = jnp.stack([jnp.asarray(base), jnp.asarray(base + 0.05 * perturbation)], axis=0)
    if float64:
        means = means.astype(jnp.complex128)
    noise = jnp.full(IMAGE_SIZE, NOISE_VARIANCE, dtype=jnp.float64 if float64 else jnp.float32)
    layouts = [_layout(ROWS_BY_CLASS[0], seed=17), _layout(ROWS_BY_CLASS[1], seed=29)]
    return means, noise, layouts


def _run(*, segmented, float64, accumulate_noise=True, reconstruct_significant_only=False):
    means, noise, layouts = _fixture(float64)
    dataset = MockDataset(N_IMAGES, np.random.default_rng(11))
    kwargs = dict(
        image_batch_size=3,
        rotation_block_size=4,
        current_size=None,
        # The production InitialModel option set apart from the reconstruction
        # selection, which each test chooses: running only with
        # reconstruct_significant_only=False hid a route that packs surviving rows
        # across images and, before it was made class-aware, accumulated every class
        # into one volume on the real path while these tests passed.
        reconstruct_significant_only=reconstruct_significant_only,
        adaptive_fraction=0.999,
        stats_use_reconstruction_probs=True,
        unweighted_high_shell_image_power=True,
        accumulate_noise=accumulate_noise,
        return_best_pose_details=True,
    )
    if float64:
        kwargs.update(use_float64_scoring=True, use_float64_normalization=True, use_float64_projections=True)
    return run_local_k_class_em(
        dataset,
        means,
        noise,
        layouts,
        "linear_interp",
        class_log_priors=np.log(np.asarray([0.45, 0.55], dtype=np.float64)),
        segmented_class_rows=segmented,
        **kwargs,
    )


def _assert_published_dtypes_match(segmented, baseline, expected_real):
    """The result contract includes its precision; a joint pass must not promote it.

    The per-class route hands each M-step call the joint normalizer, which the engine
    casts to the scoring dtype, so its posteriors, accumulators and statistics stay at
    that precision. A pass that normalizes jointly computes the normalizer in the
    accumulation dtype and would publish promoted arrays unless it applies the same
    boundary.
    """
    expected_complex = np.complex64 if expected_real is np.float32 else np.complex128
    for name in ("Ft_y", "Ft_ctf"):
        assert np.asarray(getattr(segmented, name)).dtype == expected_complex, name
        assert np.asarray(getattr(segmented, name)).dtype == np.asarray(getattr(baseline, name)).dtype, name
    for name in ("class_responsibilities", "class_posterior_sums"):
        assert np.asarray(getattr(segmented, name)).dtype == expected_real, name
        assert np.asarray(getattr(segmented, name)).dtype == np.asarray(getattr(baseline, name)).dtype, name
    for stats_name in ("log_evidence_per_image", "best_log_score_per_image", "max_posterior_per_image"):
        got = np.asarray(getattr(segmented.stats, stats_name)).dtype
        assert got == expected_real, (stats_name, got)
        assert got == np.asarray(getattr(baseline.stats, stats_name)).dtype, stats_name
        for class_index in range(2):
            per_class = np.asarray(getattr(segmented.per_class_stats[class_index], stats_name)).dtype
            assert per_class == np.asarray(getattr(baseline.per_class_stats[class_index], stats_name)).dtype
            assert per_class == expected_real, (stats_name, class_index, per_class)
    for class_index in range(2):
        assert (
            np.asarray(segmented.per_class_stats[class_index].rotation_posterior_sums).dtype
            == np.asarray(baseline.per_class_stats[class_index].rotation_posterior_sums).dtype
        )
    for field in ("wsum_sigma2_noise", "wsum_img_power"):
        got = np.asarray(getattr(segmented.aggregate_noise_stats, field)).dtype
        assert got == expected_real, (field, got)
        assert got == np.asarray(getattr(baseline.aggregate_noise_stats, field)).dtype, field


def _assert_sparse_reconstruction_route_was_used(result):
    """Fail loudly if the fixture silently took the dense adjoint instead.

    The significant-only route is the one that packs surviving rows and scatters them
    outside the bucket program; it is selected only when the local support is large
    enough. A fixture below that size tests the other route and proves nothing about
    this one.
    """
    profile = result.profile_summary or {}
    chunks = int(profile.get("sparse_adjoint_chunk_count", 0))
    assert chunks > 0, f"the significant-only reconstruction route was not exercised: {chunks} sparse chunks"


def _assert_every_class_is_populated(result):
    """Guard the comparison itself: a collapsed class would make it vacuous."""
    responsibilities = np.asarray(result.class_responsibilities, dtype=np.float64)
    masses = np.asarray(result.class_posterior_sums, dtype=np.float64)
    assert responsibilities.shape == (2, N_IMAGES)
    assert masses.min() > 0.5 * N_IMAGES / 10.0, masses
    assert responsibilities.min() > 1e-3, responsibilities
    for class_index in range(2):
        for name in ("Ft_y", "Ft_ctf"):
            values = np.abs(np.asarray(getattr(result, name)[class_index]))
            assert values.max() > 0.0 and np.count_nonzero(values) > values.size // 10, (name, class_index)
        angular = np.asarray(result.per_class_stats[class_index].rotation_posterior_sums, dtype=np.float64)
        assert angular.sum() > 0.1, (class_index, angular)


@pytest.mark.parametrize(
    "float64, rtol, atol_scale",
    [(False, 1e-6, 1e-6), (True, 1e-12, 1e-12)],
    ids=["float32", "float64"],
)
def test_segmented_pass_matches_per_class_calls(float64, rtol, atol_scale):
    recorded = []
    original = local_big_jit._class_segment_statistics

    def record_dtypes(probs, scores, reconstruction_probs, **kwargs):
        recorded.append(str(scores.dtype))
        return original(probs, scores, reconstruction_probs, **kwargs)

    baseline = _run(segmented=False, float64=float64)
    local_big_jit._class_segment_statistics = record_dtypes
    try:
        segmented = _run(segmented=True, float64=float64)
    finally:
        local_big_jit._class_segment_statistics = original

    # Measure the precision path rather than inferring it from output dtypes.
    assert recorded, "the segmented pass did not reduce class segments"
    assert set(recorded) == {"float64" if float64 else "float32"}, recorded
    _assert_every_class_is_populated(baseline)
    _assert_every_class_is_populated(segmented)
    _assert_published_dtypes_match(segmented, baseline, np.float64 if float64 else np.float32)

    def close(got, want, label):
        got = np.asarray(got)
        want = np.asarray(want)
        scale = float(np.abs(want).max()) if want.size else 1.0
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol_scale * max(scale, 1e-30), err_msg=label)

    for class_index in range(2):
        close(segmented.Ft_y[class_index], baseline.Ft_y[class_index], f"Ft_y class {class_index}")
        close(segmented.Ft_ctf[class_index], baseline.Ft_ctf[class_index], f"Ft_ctf class {class_index}")
        close(
            segmented.per_class_stats[class_index].rotation_posterior_sums,
            baseline.per_class_stats[class_index].rotation_posterior_sums,
            f"angular posterior class {class_index}",
        )
        close(
            segmented.per_class_stats[class_index].log_evidence_per_image,
            baseline.per_class_stats[class_index].log_evidence_per_image,
            f"class {class_index} normalizer",
        )
    close(segmented.class_responsibilities, baseline.class_responsibilities, "class responsibilities")
    close(segmented.class_posterior_sums, baseline.class_posterior_sums, "class posterior sums")
    close(segmented.stats.log_evidence_per_image, baseline.stats.log_evidence_per_image, "joint log evidence")
    for field in ("wsum_sigma2_noise", "wsum_img_power"):
        close(
            getattr(segmented.aggregate_noise_stats, field),
            getattr(baseline.aggregate_noise_stats, field),
            f"aggregate noise {field}",
        )
    close([segmented.aggregate_noise_stats.sumw], [baseline.aggregate_noise_stats.sumw], "aggregate noise sumw")

    # Discrete decisions and published metadata must agree exactly.
    np.testing.assert_array_equal(
        np.asarray(segmented.per_class_hard_assignments), np.asarray(baseline.per_class_hard_assignments),
    )
    np.testing.assert_array_equal(np.asarray(segmented.class_assignments), np.asarray(baseline.class_assignments))
    np.testing.assert_array_equal(np.asarray(segmented.pose_assignments), np.asarray(baseline.pose_assignments))
    for class_index in range(2):
        np.testing.assert_array_equal(
            np.asarray(segmented.per_class_best_pose_rotation_ids[class_index]),
            np.asarray(baseline.per_class_best_pose_rotation_ids[class_index]),
        )


def test_segmented_pass_publishes_the_canonical_source_eulers():
    """Source angles are published, not reconstructed from the winning rotation."""
    baseline = _run(segmented=False, float64=False)
    segmented = _run(segmented=True, float64=False)

    assert baseline.per_class_best_pose_eulers_deg is not None
    assert segmented.per_class_best_pose_eulers_deg is not None, "segmented execution dropped source Eulers"
    assert baseline.best_pose_eulers_deg is not None and segmented.best_pose_eulers_deg is not None
    for class_index in range(2):
        got = np.asarray(segmented.per_class_best_pose_eulers_deg[class_index], dtype=np.float64)
        want = np.asarray(baseline.per_class_best_pose_eulers_deg[class_index], dtype=np.float64)
        np.testing.assert_array_equal(got, want)
        # The published values are the layout's canonical angles, not matrix-derived ones.
        np.testing.assert_array_equal(got[:, :2], np.tile(CANONICAL_EULERS[:2], (N_IMAGES, 1)))
    np.testing.assert_array_equal(
        np.asarray(segmented.best_pose_eulers_deg, dtype=np.float64),
        np.asarray(baseline.best_pose_eulers_deg, dtype=np.float64),
    )


def test_segmented_rows_refuse_external_normalization():
    means, noise, layouts = _fixture(False)
    dataset = MockDataset(N_IMAGES, np.random.default_rng(17))
    with pytest.raises(NotImplementedError, match="external normalization_log_evidence"):
        run_local_k_class_em(
            dataset, means, noise, layouts, "linear_interp",
            class_log_priors=np.zeros(2), image_batch_size=3, rotation_block_size=4, current_size=None,
            normalization_log_evidence=np.zeros(N_IMAGES), segmented_class_rows=True,
        )



def test_class_packs_contain_only_their_own_rows():
    """Row identities, not aggregate weights: each class's pack is exactly its segment.

    The significant-only route packs surviving rows and scatters them outside the
    bucket program; with one joint pack every class landed in volume 0. Assert the
    property that failed, exactly and without tolerances: every packed row index of
    class k lies in class k's segment, the packs are disjoint, and together they hold
    precisely the rows a single joint pack would have selected.
    """
    from recovar.em.local.local_bucket_stages import (
        _build_nonzero_reconstruction_pack_indices,
        build_class_segment_reconstruction_packs,
    )

    rng = np.random.default_rng(131)
    n_images, n_classes, seg, n_trans = 4, 3, 5, 2
    rows = n_classes * seg
    local_mask = rng.random((n_images, rows)) > 0.25
    significant = rng.random((n_images, rows)) > 0.4
    probs_sum_t = np.where(rng.random((n_images, rows)) > 0.3, rng.random((n_images, rows)), 0.0)

    packs = build_class_segment_reconstruction_packs(
        significant, local_mask, probs_sum_t, rotation_block_size=16,
        n_classes=n_classes, segment_rotation_count=seg,
    )
    assert len(packs) == n_classes
    selected_by_class = []
    for class_index, (take, mask, counts, row_count) in enumerate(packs):
        start_row, stop_row = class_index * seg, (class_index + 1) * seg
        chosen = take[mask]
        assert chosen.size == int(np.asarray(counts).sum()) == row_count
        # Every packed row belongs to this class's segment, exactly.
        assert chosen.size == 0 or (chosen.min() >= start_row and chosen.max() < stop_row), (class_index, chosen)
        # Per image, the packed rows are the surviving rows of this segment.
        for image in range(n_images):
            expected = np.flatnonzero(
                significant[image, start_row:stop_row] & local_mask[image, start_row:stop_row]
                & (probs_sum_t[image, start_row:stop_row] > 0.0)
            ) + start_row
            np.testing.assert_array_equal(np.sort(take[image][mask[image]]), expected)
        selected_by_class.append(set(chosen.tolist()))

    # Disjoint, and together exactly the joint selection.
    for a in range(n_classes):
        for b in range(a + 1, n_classes):
            assert not (selected_by_class[a] & selected_by_class[b])
    joint_take, joint_mask, _counts, _rows = _build_nonzero_reconstruction_pack_indices(
        significant, local_mask, probs_sum_t, rotation_block_size=16,
    )
    joint_rows = {
        (image, int(row))
        for image in range(n_images)
        for row in joint_take[image][joint_mask[image]]
    }
    class_rows = {
        (image, int(row))
        for class_index, (take, mask, _c, _r) in enumerate(packs)
        for image in range(n_images)
        for row in take[image][mask[image]]
    }
    assert class_rows == joint_rows


@pytest.mark.parametrize("float64", [False, True], ids=["float32", "float64"])
@pytest.mark.parametrize("empty_class", [0, 1])
def test_a_class_without_rows_receives_an_exactly_zero_volume(float64, empty_class):
    """An empty class must receive nothing at all, in either precision.

    Both classes are exercised deliberately. Emptying only class 1 cannot detect the
    bug this guards: when every class's rows were scattered into volume 0, class 1
    came back zero anyway and such a test passed. Emptying class 0 while class 1 is
    populated fails against that implementation, because volume 0 would hold class
    1's rows.
    """
    means, noise, layouts = _fixture(float64)
    empty = layouts[empty_class]
    emptied = type(empty)(**{**empty.__dict__, "sample_mask_flat": np.zeros_like(empty.sample_mask_flat)})
    layouts = [emptied if index == empty_class else layout for index, layout in enumerate(layouts)]
    dataset = MockDataset(N_IMAGES, np.random.default_rng(11))
    kwargs = dict(
        image_batch_size=3, rotation_block_size=4, current_size=None,
        reconstruct_significant_only=True, adaptive_fraction=0.999,
        stats_use_reconstruction_probs=True, unweighted_high_shell_image_power=True,
        accumulate_noise=True, return_best_pose_details=True,
        # The profile carries the sparse chunk count, which is how this test proves it
        # reached the significant-only scatter rather than the dense adjoint.
        return_profile=True,
    )
    if float64:
        kwargs.update(use_float64_scoring=True, use_float64_normalization=True, use_float64_projections=True)
    result = run_local_k_class_em(
        dataset, means, noise, layouts, "linear_interp",
        class_log_priors=np.log(np.asarray([0.5, 0.5], dtype=np.float64)),
        segmented_class_rows=True, **kwargs,
    )
    _assert_sparse_reconstruction_route_was_used(result)
    empty_y = np.asarray(result.Ft_y[empty_class])
    empty_ctf = np.asarray(result.Ft_ctf[empty_class])
    np.testing.assert_array_equal(empty_y, np.zeros_like(empty_y))
    np.testing.assert_array_equal(empty_ctf, np.zeros_like(empty_ctf))
    populated = np.asarray(result.Ft_ctf[1 - empty_class])
    assert np.abs(populated).max() > 0.0
