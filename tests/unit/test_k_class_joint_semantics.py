import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.em.dense_single_volume.k_class as k_class_module
from recovar.em.dense_single_volume.helpers.orientation_priors import (
    class_weights_from_direction_prior,
    normalize_class_direction_prior_per_half,
)
from recovar.em.dense_single_volume.helpers.types import make_noise_stats, make_relion_stats
from recovar.em.dense_single_volume.firstiter_cc import (
    _safe_dense_k_class_rotation_block_size,
    _safe_firstiter_cc_image_batch_size,
)
from recovar.em.dense_single_volume.k_class import (
    _ClassFineGridSignificanceMask,
    _assemble_result,
    _build_fine_grid_significance_mask,
    _dense_engine_kwargs_for_class,
    run_dense_k_class_em,
    run_dense_k_class_em_adaptive,
    run_local_k_class_em,
)
from recovar.em.dense_single_volume.iteration_loop import _build_firstiter_cc_pass2_grids
from recovar.em.dense_single_volume.local_layout import LocalHypothesisLayout
from recovar.em.sampling import read_relion_direction_priors


def _stats(log_evidence, best_score, pmax, n_rot=3):
    return make_relion_stats(
        log_evidence_per_image=np.asarray(log_evidence, dtype=np.float32),
        best_log_score_per_image=np.asarray(best_score, dtype=np.float32),
        max_posterior_per_image=np.asarray(pmax, dtype=np.float32),
        rotation_posterior_sums=np.zeros(n_rot, dtype=np.float32),
    )


def test_k_class_hard_assignment_uses_joint_best_pose_not_marginal_class():
    result = _assemble_result(
        class_log_evidence=np.asarray([[0.0], [-0.2]], dtype=np.float64),
        new_means=[jnp.zeros(2), jnp.zeros(2)],
        Ft_y=[jnp.zeros(2), jnp.zeros(2)],
        Ft_ctf=[jnp.zeros(2), jnp.zeros(2)],
        per_class_hard_assignments=np.asarray([[3], [7]], dtype=np.int32),
        per_class_stats=(
            _stats([0.0], [-5.0], [0.01]),
            _stats([-0.2], [-1.0], [0.20]),
        ),
        noise_stats=None,
    )

    assert np.asarray(result.class_responsibilities)[0, 0] > np.asarray(result.class_responsibilities)[1, 0]
    np.testing.assert_array_equal(np.asarray(result.class_assignments), np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(result.pose_assignments), np.asarray([7], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(result.stats.max_posterior_per_image), np.asarray([0.20], dtype=np.float32))


def test_k_class_noise_aggregate_uses_joint_relion_sum_weight():
    """Class3D sigma-offset denominator is RELION's global sum_weight, not K*N.

    RELION accumulates ``wsum_model.sigma2_offset`` over the joint class x pose
    posterior and updates ``sigma2_offset = wsum / (2 * sum_weight)``.  A
    per-class RECOVAR engine reports ``sumw == n_images`` for each class, so
    aggregating those bookkeeping values would over-count by ``n_classes``.
    """

    noise_stats = (
        make_noise_stats(
            wsum_sigma2_noise=np.asarray([1.0], dtype=np.float32),
            wsum_img_power=np.asarray([2.0], dtype=np.float32),
            wsum_sigma2_offset=30.0,
            sumw=2.0,
        ),
        make_noise_stats(
            wsum_sigma2_noise=np.asarray([3.0], dtype=np.float32),
            wsum_img_power=np.asarray([4.0], dtype=np.float32),
            wsum_sigma2_offset=50.0,
            sumw=2.0,
        ),
    )

    result = _assemble_result(
        class_log_evidence=np.asarray(
            [
                [np.log(0.25), np.log(0.75)],
                [np.log(0.75), np.log(0.25)],
            ],
            dtype=np.float64,
        ),
        new_means=[jnp.zeros(1), jnp.zeros(1)],
        Ft_y=[jnp.zeros(1), jnp.zeros(1)],
        Ft_ctf=[jnp.zeros(1), jnp.zeros(1)],
        per_class_hard_assignments=np.zeros((2, 2), dtype=np.int32),
        per_class_stats=(
            _stats([0.0, 0.0], [0.0, 0.0], [0.5, 0.5], n_rot=1),
            _stats([0.0, 0.0], [0.0, 0.0], [0.5, 0.5], n_rot=1),
        ),
        noise_stats=noise_stats,
    )

    assert result.aggregate_noise_stats is not None
    assert result.aggregate_noise_stats.wsum_sigma2_offset == pytest.approx(80.0)
    assert result.aggregate_noise_stats.sumw == pytest.approx(2.0)
    np.testing.assert_allclose(
        np.asarray(result.aggregate_noise_stats.wsum_img_power),
        np.asarray([3.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.aggregate_noise_stats.wsum_sigma2_offset / (2.0 * result.aggregate_noise_stats.sumw) == pytest.approx(
        20.0,
    )


def test_dense_k_class_selects_class_rotation_log_prior(monkeypatch):
    calls = []

    class TinyDataset:
        n_images = 2

    def fake_run_em(_dataset, mean, _mean_variance, _noise_variance, rotations, _translations, _disc_type, **kwargs):
        calls.append(kwargs)
        n_images = TinyDataset.n_images
        stats = make_relion_stats(
            log_evidence_per_image=np.full(n_images, float(len(calls)), dtype=np.float32),
            best_log_score_per_image=np.full(n_images, float(len(calls)), dtype=np.float32),
            max_posterior_per_image=np.full(n_images, 0.5, dtype=np.float32),
            rotation_posterior_sums=np.zeros(rotations.shape[0], dtype=np.float32),
        )
        return (
            jnp.zeros_like(mean),
            np.zeros(n_images, dtype=np.int32),
            jnp.zeros_like(mean),
            jnp.zeros_like(mean),
            stats,
        )

    monkeypatch.setattr(k_class_module, "run_em", fake_run_em)

    class_rotation_log_prior = np.asarray(
        [
            [0.0, -1.0, -2.0],
            [-3.0, -4.0, -5.0],
        ],
        dtype=np.float32,
    )
    run_dense_k_class_em(
        TinyDataset(),
        jnp.zeros((2, 4), dtype=jnp.complex64),
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(4, dtype=jnp.float32),
        np.zeros((3, 3, 3), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        "linear_interp",
        class_rotation_log_prior=class_rotation_log_prior,
    )

    assert len(calls) == 4
    for call_index, expected_class in enumerate([0, 1, 0, 1]):
        np.testing.assert_allclose(
            np.asarray(calls[call_index]["rotation_log_prior"]),
            class_rotation_log_prior[expected_class],
        )
        assert "class_rotation_log_prior" not in calls[call_index]


def test_dense_k_class_decodes_best_pose_details(monkeypatch):
    calls = []

    class TinyDataset:
        n_images = 2

    rotations = np.asarray(
        [
            np.eye(3, dtype=np.float32),
            np.diag([1.0, -1.0, -1.0]).astype(np.float32),
            np.diag([-1.0, 1.0, -1.0]).astype(np.float32),
        ],
    )
    translations = np.asarray([[-2.0, 0.0], [3.0, 4.0]], dtype=np.float32)

    def fake_run_em(_dataset, mean, _mean_variance, _noise_variance, rotations_arg, _translations, _disc_type, **kwargs):
        calls.append(kwargs)
        n_images = TinyDataset.n_images
        final_call = len(calls) > 2
        class_index = (len(calls) - 1) % 2
        if final_call and class_index == 0:
            hard_assignment = np.asarray([0, 3], dtype=np.int32)
            best_score = np.asarray([10.0, 20.0], dtype=np.float32)
        elif final_call:
            hard_assignment = np.asarray([5, 2], dtype=np.int32)
            best_score = np.asarray([30.0, 15.0], dtype=np.float32)
        else:
            hard_assignment = np.zeros(n_images, dtype=np.int32)
            best_score = np.zeros(n_images, dtype=np.float32)
        stats = make_relion_stats(
            log_evidence_per_image=np.zeros(n_images, dtype=np.float32),
            best_log_score_per_image=best_score,
            max_posterior_per_image=np.ones(n_images, dtype=np.float32),
            rotation_posterior_sums=np.zeros(rotations_arg.shape[0], dtype=np.float32),
        )
        return (
            jnp.zeros_like(mean),
            hard_assignment,
            jnp.zeros_like(mean),
            jnp.zeros_like(mean),
            stats,
        )

    monkeypatch.setattr(k_class_module, "run_em", fake_run_em)

    result = run_dense_k_class_em(
        TinyDataset(),
        jnp.zeros((2, 4), dtype=jnp.complex64),
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(4, dtype=jnp.float32),
        rotations,
        translations,
        "linear_interp",
        return_best_pose_details=True,
    )

    np.testing.assert_array_equal(np.asarray(result.class_assignments), np.asarray([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(result.pose_assignments), np.asarray([5, 3], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(result.best_pose_rotation_ids), np.asarray([2, 1], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(result.best_pose_rotations), rotations[[2, 1]])
    np.testing.assert_allclose(np.asarray(result.best_pose_translations), translations[[1, 1]])


def test_dense_k_class_single_class_skips_score_probe(monkeypatch):
    calls = []

    class TinyDataset:
        n_images = 2

    rotations = np.asarray(
        [
            np.eye(3, dtype=np.float32),
            np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        ],
    )
    translations = np.asarray([[-2.0, 0.0], [3.0, 4.0]], dtype=np.float32)

    def fake_run_em(_dataset, mean, _mean_variance, _noise_variance, rotations_arg, _translations, _disc_type, **kwargs):
        calls.append(kwargs)
        hard_assignment = np.asarray([1, 2], dtype=np.int32)
        stats = make_relion_stats(
            log_evidence_per_image=np.asarray([2.0, 3.0], dtype=np.float32),
            best_log_score_per_image=np.asarray([1.5, 2.5], dtype=np.float32),
            max_posterior_per_image=np.asarray([0.25, 0.75], dtype=np.float32),
            rotation_posterior_sums=np.arange(rotations_arg.shape[0], dtype=np.float32),
        )
        return (
            jnp.zeros_like(mean),
            hard_assignment,
            jnp.ones_like(mean),
            jnp.ones_like(mean) * 2,
            stats,
        )

    monkeypatch.setattr(k_class_module, "run_em", fake_run_em)

    result = run_dense_k_class_em(
        TinyDataset(),
        jnp.zeros((1, 4), dtype=jnp.complex64),
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(4, dtype=jnp.float32),
        rotations,
        translations,
        "linear_interp",
        return_best_pose_details=True,
    )

    assert len(calls) == 1
    assert calls[0]["return_stats"] is True
    assert calls[0]["accumulate_noise"] is False
    assert "normalization_log_evidence" not in calls[0]
    np.testing.assert_array_equal(np.asarray(result.class_assignments), np.asarray([0, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(result.pose_assignments), np.asarray([1, 2], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(result.class_responsibilities), np.ones((1, 2), dtype=np.float32))
    np.testing.assert_allclose(np.asarray(result.class_posterior_sums), np.asarray([2.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(result.best_pose_rotation_ids), np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(result.best_pose_translations), translations[[1, 0]])


def test_dense_k_class_wta_mask_skips_score_probe(monkeypatch):
    calls = []

    class TinyDataset:
        n_images = 2

    def fake_run_em(
        _dataset,
        mean,
        _mean_variance,
        _noise_variance,
        rotations_arg,
        _translations,
        _disc_type,
        **kwargs,
    ):
        calls.append(kwargs)
        n_images = TinyDataset.n_images
        n_rot = int(np.asarray(rotations_arg).shape[0])
        stats = make_relion_stats(
            log_evidence_per_image=np.full(n_images, float(len(calls)), dtype=np.float32),
            best_log_score_per_image=np.full(n_images, float(len(calls)), dtype=np.float32),
            max_posterior_per_image=np.ones(n_images, dtype=np.float32),
            rotation_posterior_sums=np.zeros(n_rot, dtype=np.float32),
        )
        return (
            jnp.zeros_like(mean),
            np.zeros(n_images, dtype=np.int32),
            jnp.zeros_like(mean),
            jnp.zeros_like(mean),
            stats,
        )

    monkeypatch.setattr(k_class_module, "run_em", fake_run_em)

    run_dense_k_class_em(
        TinyDataset(),
        jnp.zeros((2, 4), dtype=jnp.complex64),
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(4, dtype=jnp.float32),
        np.zeros((3, 3, 3), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        "linear_interp",
        relion_firstiter_winner_take_all=True,
        class_rotation_translation_mask=np.ones((2, TinyDataset.n_images, 3, 1), dtype=bool),
    )

    assert len(calls) == 2
    assert all("normalization_log_evidence" not in call for call in calls)


def test_adaptive_k_class_firstiter_override_redecodes_best_pose_details(monkeypatch):
    score_calls = []

    class TinyDataset:
        n_units = 2
        n_images = 2
        image_shape = (8, 8)
        volume_shape = (4, 4, 4)
        dtype = jnp.complex64

    fine_rotations = np.asarray(
        [
            np.eye(3, dtype=np.float32),
            np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        ],
    )
    fine_translations = np.asarray([[-1.0, 0.0], [2.0, 0.0]], dtype=np.float32)

    def pose_details(hard):
        hard = np.asarray(hard, dtype=np.int64)
        rot_ids = hard // fine_translations.shape[0]
        trans_ids = hard % fine_translations.shape[0]
        return fine_rotations[rot_ids], fine_translations[trans_ids], rot_ids.astype(np.int32)

    def fake_run_em(
        _dataset,
        mean,
        _mean_variance,
        _noise_variance,
        rotations_arg,
        _translations,
        _disc_type,
        **kwargs,
    ):
        class_index = 0 if float(np.asarray(mean)[0].real) == 10.0 else 1
        image_indices = kwargs.get("image_indices")
        is_fine = image_indices is not None
        score_calls.append({"class_index": class_index, "is_fine": is_fine, **kwargs})
        n_images = len(image_indices) if is_fine else TinyDataset.n_images
        n_rot = int(np.asarray(rotations_arg).shape[0])
        if is_fine:
            hard = (np.asarray([1], dtype=np.int32), np.asarray([2], dtype=np.int32))[class_index]
            scores = (np.asarray([9.0], dtype=np.float32), np.asarray([8.0], dtype=np.float32))
        else:
            hard = (np.asarray([0, 0], dtype=np.int32), np.asarray([1, 1], dtype=np.int32))[class_index]
            scores = (np.asarray([1.0, 5.0], dtype=np.float32), np.asarray([5.0, 1.0], dtype=np.float32))
        stats = make_relion_stats(
            log_evidence_per_image=np.zeros(n_images, dtype=np.float32),
            best_log_score_per_image=scores[class_index],
            max_posterior_per_image=np.ones(n_images, dtype=np.float32),
            rotation_posterior_sums=np.zeros(n_rot, dtype=np.float32),
        )
        return (
            jnp.zeros_like(mean),
            hard,
            jnp.zeros_like(mean),
            jnp.zeros_like(mean),
            stats,
        )

    monkeypatch.setattr(k_class_module, "run_em", fake_run_em)

    result = run_dense_k_class_em_adaptive(
        TinyDataset(),
        jnp.stack(
            [
                jnp.full(4, 10.0, dtype=jnp.complex64),
                jnp.full(4, 20.0, dtype=jnp.complex64),
            ],
            axis=0,
        ),
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        np.repeat(np.eye(3, dtype=np.float32)[None], 1, axis=0),
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        fine_rotations,
        fine_translations,
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
        "linear_interp",
        firstiter_cc_pass2_only_best_coarse=True,
        skip_significance_pruning=True,
        return_best_pose_details=True,
    )

    fine_calls = [call for call in score_calls if call["is_fine"]]
    assert [call["class_index"] for call in fine_calls] == [0, 1]
    np.testing.assert_array_equal(fine_calls[0]["image_indices"], np.asarray([1], dtype=np.int64))
    np.testing.assert_array_equal(fine_calls[1]["image_indices"], np.asarray([0], dtype=np.int64))
    np.testing.assert_array_equal(np.asarray(result.class_assignments), np.asarray([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(result.pose_assignments), np.asarray([2, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(result.best_pose_rotation_ids), np.asarray([1, 0], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(result.best_pose_translations), fine_translations[[0, 1]])


def test_adaptive_k_class_firstiter_uses_coarse_current_size_for_probe(monkeypatch):
    score_calls = []

    class TinyDataset:
        n_units = 2
        n_images = 2
        image_shape = (8, 8)
        volume_shape = (4, 4, 4)
        dtype = jnp.complex64

    def fake_run_em(
        _dataset,
        mean,
        _mean_variance,
        _noise_variance,
        rotations,
        _translations,
        _disc_type,
        **kwargs,
    ):
        class_index = 0 if float(np.asarray(mean)[0].real) == 10.0 else 1
        image_indices = kwargs.get("image_indices")
        is_fine = image_indices is not None
        n_images = len(image_indices) if is_fine else TinyDataset.n_images
        n_rot = int(np.asarray(rotations).shape[0])
        score_calls.append({"class_index": class_index, "is_fine": is_fine, **kwargs})
        best_score = (
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
        )[class_index]
        if is_fine:
            best_score = np.ones(n_images, dtype=np.float32)
        stats = make_relion_stats(
            log_evidence_per_image=np.zeros(n_images, dtype=np.float32),
            best_log_score_per_image=best_score,
            max_posterior_per_image=np.ones(n_images, dtype=np.float32),
            rotation_posterior_sums=np.zeros(n_rot, dtype=np.float32),
        )
        return (
            jnp.zeros_like(mean),
            np.zeros(n_images, dtype=np.int32),
            jnp.zeros_like(mean),
            jnp.zeros_like(mean),
            stats,
        )

    monkeypatch.setattr(k_class_module, "run_em", fake_run_em)

    run_dense_k_class_em_adaptive(
        TinyDataset(),
        jnp.stack(
            [
                jnp.full(4, 10.0, dtype=jnp.complex64),
                jnp.full(4, 20.0, dtype=jnp.complex64),
            ],
            axis=0,
        ),
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        np.zeros((2, 3, 3), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((4, 3, 3), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        "linear_interp",
        coarse_current_size=26,
        fine_current_size=56,
        current_size=56,
        firstiter_cc_pass2_only_best_coarse=True,
        fine_rotation_block_size=123,
    )

    coarse_calls = [call for call in score_calls if not call["is_fine"]]
    fine_calls = [call for call in score_calls if call["is_fine"]]
    assert len(coarse_calls) == 2
    assert len(fine_calls) == 2
    assert {call["current_size"] for call in coarse_calls} == {26}
    assert {call["current_size"] for call in fine_calls} == {56}
    assert {call["rotation_block_size"] for call in fine_calls} == {123}


def test_lazy_k_class_adaptive_mask_matches_dense_blocks_without_materializing():
    n_classes = 2
    n_images = 3
    n_rot_coarse = 3
    n_trans_coarse = 2
    rot_parent_map = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    trans_parent_map = np.asarray([0, 0, 1], dtype=np.int64)
    n_rot_fine = int(rot_parent_map.size)
    n_trans_fine = int(trans_parent_map.size)
    significant_by_class = [
        [
            np.asarray([0, 3], dtype=np.int32),
            None,
            np.asarray([5], dtype=np.int32),
        ],
        [
            np.asarray([1], dtype=np.int32),
            np.asarray([2, 4], dtype=np.int32),
            None,
        ],
    ]
    global_winner = np.asarray([0, 1, 0], dtype=np.int64)
    lazy = _ClassFineGridSignificanceMask(
        significant_sample_indices_by_class=significant_by_class,
        n_rot_coarse=n_rot_coarse,
        n_trans_coarse=n_trans_coarse,
        n_rot_fine=n_rot_fine,
        n_trans_fine=n_trans_fine,
        rot_parent_map=rot_parent_map,
        trans_parent_map=trans_parent_map,
        n_images=n_images,
        n_classes=n_classes,
        global_winner=global_winner,
    )

    with pytest.raises(TypeError):
        np.asarray(lazy)
    selected = _dense_engine_kwargs_for_class(
        {"class_rotation_translation_mask": lazy},
        class_index=1,
        n_classes=n_classes,
    )["rotation_translation_mask"]
    with pytest.raises(TypeError):
        np.asarray(selected)

    for class_index in range(n_classes):
        dense = _build_fine_grid_significance_mask(
            significant_by_class[class_index],
            n_rot_coarse=n_rot_coarse,
            n_trans_coarse=n_trans_coarse,
            n_rot_fine=n_rot_fine,
            n_trans_fine=n_trans_fine,
            rot_oversampling_factor=2,
            trans_oversampling_factor=2,
            rot_parent_map=rot_parent_map,
            trans_parent_map=trans_parent_map,
            n_images=n_images,
        )
        dense[global_winner != class_index, :, :] = False
        per_class = lazy.for_class(class_index)
        for start, end, r0, rotation_block_size, batch_count in (
            (0, 2, 1, 4, 4),
            (2, 3, 4, 4, 2),
        ):
            actual_rot = max(0, min(rotation_block_size, n_rot_fine - r0))
            expected = np.zeros((batch_count, rotation_block_size, n_trans_fine), dtype=bool)
            expected[: end - start, :actual_rot, :] = dense[start:end, r0 : r0 + actual_rot, :]
            actual = per_class.block_mask(
                r0=r0,
                r1=r0 + rotation_block_size,
                start=start,
                end=end,
                batch_count=batch_count,
                rotation_block_size=rotation_block_size,
            )
            np.testing.assert_array_equal(np.asarray(actual), expected)
            assert per_class.block_has_candidates(
                r0=r0,
                start=start,
                end=end,
                batch_count=batch_count,
                rotation_block_size=rotation_block_size,
            ) == bool(expected.any())


def test_firstiter_adaptive_translation_perturbation_uses_coarse_step():
    """RELION perturbs oversampled translations by random_perturbation * offset_step.

    Source: HealpixSampling::getTranslations first subdivides each translation
    cell, then adds ``random_perturbation * offset_step / pixel_size`` to every
    oversampled child. The perturbation does not use the subdivided fine step.
    """

    coarse_rot = np.eye(3, dtype=np.float32)[None]
    coarse_trans = np.array([[0.0, 0.0]], dtype=np.float32)
    random_perturbation = 0.25
    translation_step_px = 2.0

    (
        _coarse_rot,
        _coarse_trans,
        _fine_rot,
        fine_trans,
        _rot_parent_map,
        trans_parent_map,
    ) = _build_firstiter_cc_pass2_grids(
        coarse_rot,
        coarse_trans,
        coarse_trans,
        coarse_healpix_order=0,
        adaptive_oversampling=1,
        translation_step_px=translation_step_px,
        random_perturbation=random_perturbation,
    )

    expected_shift = random_perturbation * translation_step_px
    expected = np.array(
        [
            [-0.5, -0.5],
            [-0.5, 0.5],
            [0.5, -0.5],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    ) + np.float32(expected_shift)
    np.testing.assert_allclose(fine_trans, expected, atol=1e-6)
    np.testing.assert_array_equal(trans_parent_map, np.zeros(4, dtype=np.int64))


def test_firstiter_cc_batch_cap_keeps_relion_default_batch_for_256_box():
    assert _safe_firstiter_cc_image_batch_size(116, (256, 256)) >= 50
    assert _safe_dense_k_class_rotation_block_size(116, 50) >= 1_000


def test_firstiter_coarse_score_probe_uses_larger_safe_batch(monkeypatch):
    calls = []

    class TinyDataset:
        n_units = 300
        n_images = 300
        image_shape = (256, 256)
        volume_shape = (4, 4, 4)
        dtype = jnp.complex64

    means = jnp.stack(
        [
            jnp.full(64, 10.0, dtype=jnp.complex64),
            jnp.full(64, 20.0, dtype=jnp.complex64),
        ],
        axis=0,
    )

    def fake_run_em(
        _dataset,
        mean,
        _mean_variance,
        _noise_variance,
        rotations_arg,
        _translations,
        _disc_type,
        **kwargs,
    ):
        class_index = 0 if float(np.asarray(mean)[0].real) == 10.0 else 1
        image_indices = kwargs.get("image_indices")
        is_fine = image_indices is not None
        calls.append({"class_index": class_index, "is_fine": is_fine, **kwargs})
        n_images = len(image_indices) if is_fine else TinyDataset.n_units
        best_score = np.full(n_images, 10.0 if class_index == 0 else 0.0, dtype=np.float32)
        stats = make_relion_stats(
            log_evidence_per_image=best_score,
            best_log_score_per_image=best_score,
            max_posterior_per_image=np.ones(n_images, dtype=np.float32),
            rotation_posterior_sums=np.zeros(rotations_arg.shape[0], dtype=np.float32),
        )
        return (
            jnp.zeros(64, dtype=jnp.complex64),
            np.zeros(n_images, dtype=np.int32),
            jnp.zeros(64, dtype=jnp.complex64),
            jnp.zeros(64, dtype=jnp.complex64),
            stats,
        )

    monkeypatch.setattr(k_class_module, "run_em", fake_run_em)

    run_dense_k_class_em_adaptive(
        TinyDataset(),
        means,
        jnp.ones((2, 64), dtype=jnp.float32),
        jnp.ones((2, 64), dtype=jnp.float32),
        np.zeros((2, 3, 3), dtype=np.float32),
        np.zeros((29, 2), dtype=np.float32),
        np.zeros((4, 3, 3), dtype=np.float32),
        np.zeros((116, 2), dtype=np.float32),
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.arange(116, dtype=np.int64) // 4,
        "linear_interp",
        firstiter_cc_pass2_only_best_coarse=True,
        image_batch_size=50,
        rotation_block_size=2,
        relion_firstiter_winner_take_all=True,
    )

    coarse_calls = [call for call in calls if not call["is_fine"]]
    fine_calls = [call for call in calls if call["is_fine"]]
    expected_coarse_batch = _safe_firstiter_cc_image_batch_size(29, TinyDataset.image_shape)
    assert {call["image_batch_size"] for call in coarse_calls} == {expected_coarse_batch}
    assert {call["image_batch_size"] for call in fine_calls} == {50}


def test_firstiter_fine_pass_scores_only_winner_class_subsets(monkeypatch):
    calls = []

    class TinyDataset:
        n_units = 6
        n_images = 6
        image_shape = (8, 8)
        volume_shape = (4, 4, 4)
        dtype = jnp.complex64

    means = jnp.stack(
        [
            jnp.full(64, 10.0, dtype=jnp.complex64),
            jnp.full(64, 20.0, dtype=jnp.complex64),
        ],
        axis=0,
    )
    coarse_best_by_class = (
        np.asarray([10.0, 1.0, 10.0, 1.0, 1.0, 10.0], dtype=np.float32),
        np.asarray([1.0, 10.0, 1.0, 10.0, 10.0, 1.0], dtype=np.float32),
    )

    def fake_run_em(
        _dataset,
        mean,
        _mean_variance,
        _noise_variance,
        rotations_arg,
        _translations,
        _disc_type,
        **kwargs,
    ):
        class_index = 0 if float(np.asarray(mean)[0].real) == 10.0 else 1
        image_indices = kwargs.get("image_indices")
        is_fine = int(rotations_arg.shape[0]) == 4
        calls.append(
            {
                "class_index": class_index,
                "is_fine": is_fine,
                "image_indices": None if image_indices is None else np.asarray(image_indices, dtype=np.int64),
                "rotation_translation_mask": kwargs.get("rotation_translation_mask"),
                "translation_log_prior": kwargs.get("translation_log_prior"),
            }
        )
        if is_fine:
            subset = np.asarray(image_indices, dtype=np.int64)
            hard_assignment = np.asarray([0, 1, 2], dtype=np.int32) + 3 * class_index
            hard_assignment = hard_assignment[: subset.size]
            best_score = np.full(subset.size, 100.0 + class_index, dtype=np.float32)
            n_images = subset.size
        else:
            hard_assignment = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32)
            best_score = coarse_best_by_class[class_index]
            n_images = TinyDataset.n_units
        stats = make_relion_stats(
            log_evidence_per_image=best_score,
            best_log_score_per_image=best_score,
            max_posterior_per_image=np.ones(n_images, dtype=np.float32),
            rotation_posterior_sums=np.zeros(rotations_arg.shape[0], dtype=np.float32),
        )
        return (
            jnp.zeros(64, dtype=jnp.complex64),
            hard_assignment,
            jnp.full(64, class_index + 1, dtype=jnp.complex64),
            jnp.full(64, class_index + 2, dtype=jnp.complex64),
            stats,
        )

    monkeypatch.setattr(k_class_module, "run_em", fake_run_em)

    result = run_dense_k_class_em_adaptive(
        TinyDataset(),
        means,
        jnp.ones((2, 64), dtype=jnp.float32),
        jnp.ones((2, 64), dtype=jnp.float32),
        np.zeros((2, 3, 3), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((4, 3, 3), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
        "linear_interp",
        class_log_priors=np.zeros(2, dtype=np.float32),
        firstiter_cc_pass2_only_best_coarse=True,
        image_batch_size=2,
        rotation_block_size=2,
        relion_firstiter_winner_take_all=True,
        translation_log_prior=np.arange(12, dtype=np.float32).reshape(6, 2),
    )

    fine_calls = [call for call in calls if call["is_fine"]]
    assert [call["class_index"] for call in fine_calls] == [0, 1]
    np.testing.assert_array_equal(fine_calls[0]["image_indices"], np.asarray([0, 2, 5], dtype=np.int64))
    np.testing.assert_array_equal(fine_calls[1]["image_indices"], np.asarray([1, 3, 4], dtype=np.int64))
    assert fine_calls[0]["rotation_translation_mask"].shape == (3, 4, 2)
    assert fine_calls[1]["rotation_translation_mask"].shape == (3, 4, 2)
    np.testing.assert_array_equal(
        fine_calls[0]["translation_log_prior"],
        np.arange(12, dtype=np.float32).reshape(6, 2)[[0, 2, 5]],
    )
    np.testing.assert_array_equal(
        fine_calls[1]["translation_log_prior"],
        np.arange(12, dtype=np.float32).reshape(6, 2)[[1, 3, 4]],
    )
    np.testing.assert_array_equal(np.asarray(result.class_assignments), np.asarray([0, 1, 0, 1, 1, 0]))
    np.testing.assert_array_equal(np.asarray(result.pose_assignments), np.asarray([0, 3, 1, 4, 5, 2]))
    np.testing.assert_allclose(np.asarray(result.class_posterior_sums), np.asarray([3.0, 3.0], dtype=np.float32))


def test_local_k_class_single_class_skips_score_probe(monkeypatch):
    calls = []

    class TinyDataset:
        n_units = 2

    local_layout = LocalHypothesisLayout(
        n_global_rotations=2,
        n_pixels=1,
        n_psi=2,
        rotation_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        rotation_ids_flat=np.asarray([0, 1], dtype=np.int32),
        rotations_flat=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        rotation_log_priors_flat=np.zeros(2, dtype=np.float32),
        rotation_counts=np.asarray([1, 1], dtype=np.int32),
        translation_grid=np.zeros((1, 2), dtype=np.float32),
        translation_log_priors=np.zeros((2, 1), dtype=np.float32),
        rotation_posterior_ids_flat=np.asarray([0, 1], dtype=np.int32),
        sample_mask_flat=np.ones((2, 1), dtype=bool),
    )

    def fake_run_local_em_exact(
        _dataset,
        mean,
        _mean_variance,
        _noise_variance,
        _local_layout,
        _disc_type,
        **kwargs,
    ):
        calls.append(kwargs)
        stats = make_relion_stats(
            log_evidence_per_image=np.asarray([4.0, 5.0], dtype=np.float32),
            best_log_score_per_image=np.asarray([3.0, 4.0], dtype=np.float32),
            max_posterior_per_image=np.asarray([0.5, 0.8], dtype=np.float32),
            rotation_posterior_sums=np.asarray([1.0, 2.0], dtype=np.float32),
        )
        return (
            jnp.ones_like(mean),
            jnp.ones_like(mean) * 2,
            np.asarray([1, 0], dtype=np.int32),
            np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
            np.zeros((2, 2), dtype=np.float32),
            np.asarray([1, 0], dtype=np.int32),
            stats,
        )

    monkeypatch.setattr(k_class_module, "run_local_em_exact", fake_run_local_em_exact)

    result = run_local_k_class_em(
        TinyDataset(),
        jnp.zeros((1, 4), dtype=jnp.complex64),
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(4, dtype=jnp.float32),
        local_layout,
        "linear_interp",
        return_best_pose_details=True,
    )

    assert len(calls) == 1
    assert calls[0]["return_best_pose_details"] is True
    assert calls[0]["accumulate_noise"] is False
    assert "normalization_log_evidence" not in calls[0]
    np.testing.assert_array_equal(np.asarray(result.class_assignments), np.asarray([0, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(result.pose_assignments), np.asarray([1, 0], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(result.class_responsibilities), np.ones((1, 2), dtype=np.float32))
    np.testing.assert_allclose(np.asarray(result.class_posterior_sums), np.asarray([2.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(result.best_pose_rotation_ids), np.asarray([1, 0], dtype=np.int32))


def test_local_k_class_accepts_per_class_layouts_and_external_evidence(monkeypatch):
    calls = []

    class TinyDataset:
        n_units = 2

    def layout_with_prior(prior):
        return LocalHypothesisLayout(
            n_global_rotations=2,
            n_pixels=1,
            n_psi=2,
            rotation_offsets=np.asarray([0, 1, 2], dtype=np.int64),
            rotation_ids_flat=np.asarray([0, 1], dtype=np.int32),
            rotations_flat=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
            rotation_log_priors_flat=np.asarray(prior, dtype=np.float32),
            rotation_counts=np.asarray([1, 1], dtype=np.int32),
            translation_grid=np.zeros((1, 2), dtype=np.float32),
            translation_log_priors=np.zeros((2, 1), dtype=np.float32),
            rotation_posterior_ids_flat=np.asarray([0, 1], dtype=np.int32),
            sample_mask_flat=np.ones((2, 1), dtype=bool),
        )

    def fake_run_local_em_exact(
        _dataset,
        mean,
        _mean_variance,
        _noise_variance,
        local_layout,
        _disc_type,
        **kwargs,
    ):
        calls.append((local_layout, kwargs))
        call_index = len(calls)
        stats = make_relion_stats(
            log_evidence_per_image=np.full(2, float(call_index), dtype=np.float32),
            best_log_score_per_image=np.full(2, float(call_index), dtype=np.float32),
            max_posterior_per_image=np.full(2, 0.25, dtype=np.float32),
            rotation_posterior_sums=np.zeros(2, dtype=np.float32),
        )
        return (
            jnp.zeros_like(mean),
            jnp.zeros_like(mean),
            np.zeros(2, dtype=np.int32),
            stats,
        )

    monkeypatch.setattr(k_class_module, "run_local_em_exact", fake_run_local_em_exact)

    class_log_evidence = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    normalization_log_evidence = np.logaddexp(class_log_evidence[0], class_log_evidence[1])
    run_local_k_class_em(
        TinyDataset(),
        jnp.zeros((2, 4), dtype=jnp.complex64),
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(4, dtype=jnp.float32),
        (layout_with_prior([0.0, -1.0]), layout_with_prior([-2.0, -3.0])),
        "linear_interp",
        class_log_priors=np.log(np.asarray([0.5, 0.5], dtype=np.float64)),
        class_log_evidence=class_log_evidence,
        normalization_log_evidence=normalization_log_evidence,
    )

    assert len(calls) == 2
    np.testing.assert_allclose(calls[0][0].rotation_log_priors_flat, np.asarray([0.0, -1.0]))
    np.testing.assert_allclose(calls[1][0].rotation_log_priors_flat, np.asarray([-2.0, -3.0]))
    for _layout, kwargs in calls:
        np.testing.assert_allclose(kwargs["normalization_log_evidence"], normalization_log_evidence)


def test_class_direction_prior_normalizes_relion_joint_rows():
    raw_half1 = np.asarray(
        [
            [0.20, 0.10],
            [0.15, 0.55],
        ],
        dtype=np.float32,
    )
    raw_half2 = np.asarray(
        [
            [0.12, 0.18],
            [0.21, 0.49],
        ],
        dtype=np.float32,
    )

    priors = normalize_class_direction_prior_per_half([raw_half1, raw_half2], n_classes=2)

    np.testing.assert_allclose(priors[0].sum(axis=1), np.ones(2), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(priors[1].sum(axis=1), np.ones(2), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(priors[0][0], raw_half1[0] / raw_half1[0].sum(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        class_weights_from_direction_prior([raw_half1, raw_half2], n_classes=2),
        raw_half1.sum(axis=1) / raw_half1.sum(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_read_relion_direction_priors_reads_all_classes(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    model_star = tmp_path / "model.star"
    starfile.write(
        {
            "model_pdf_orient_class_1": pd.DataFrame({"rlnOrientationDistribution": [0.2, 0.3]}),
            "model_pdf_orient_class_2": pd.DataFrame({"rlnOrientationDistribution": [0.1, 0.4]}),
        },
        model_star,
    )

    priors = read_relion_direction_priors(model_star, n_classes=2)

    np.testing.assert_allclose(
        priors,
        np.asarray([[0.2, 0.3], [0.1, 0.4]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
