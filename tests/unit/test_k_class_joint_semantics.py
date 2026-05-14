from types import SimpleNamespace

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
from recovar.em.dense_single_volume.k_class import (
    _ClassFineGridSignificanceMask,
    _assemble_result,
    _build_fine_grid_significance_mask,
    _dense_engine_kwargs_for_class,
    _run_sparse_k_class_adaptive_pass2,
    run_dense_k_class_em,
    run_dense_k_class_em_adaptive,
    run_local_k_class_em,
)
from recovar.em.dense_single_volume.iteration_loop import (
    _build_firstiter_cc_pass2_grids,
    _combine_optional_half_accumulators,
)
from recovar.em.dense_single_volume.local_layout import LocalHypothesisLayout
from recovar.em.sampling import read_relion_direction_priors


def _stats(log_evidence, best_score, pmax, n_rot=3):
    return make_relion_stats(
        log_evidence_per_image=np.asarray(log_evidence, dtype=np.float32),
        best_log_score_per_image=np.asarray(best_score, dtype=np.float32),
        max_posterior_per_image=np.asarray(pmax, dtype=np.float32),
        rotation_posterior_sums=np.zeros(n_rot, dtype=np.float32),
    )


def _firstiter_probe_result(class_assignments, per_class_hard=None, n_rot=1):
    class_assignments = np.asarray(class_assignments, dtype=np.int32)
    n_images = int(class_assignments.size)
    if per_class_hard is None:
        n_classes = int(class_assignments.max(initial=0)) + 1
        per_class_hard = np.zeros((n_classes, n_images), dtype=np.int32)
    else:
        per_class_hard = np.asarray(per_class_hard, dtype=np.int32)
        n_classes = int(per_class_hard.shape[0])
    return k_class_module._DenseKClassScoreProbeResult(
        class_log_evidence=np.zeros((n_classes, n_images), dtype=np.float64),
        per_class_hard_assignments=per_class_hard,
        per_class_stats=tuple(
            make_relion_stats(
                log_evidence_per_image=np.zeros(n_images, dtype=np.float32),
                best_log_score_per_image=np.zeros(n_images, dtype=np.float32),
                max_posterior_per_image=np.ones(n_images, dtype=np.float32),
                rotation_posterior_sums=np.zeros(n_rot, dtype=np.float32),
            )
            for _ in range(n_classes)
        ),
        class_assignments=class_assignments,
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


def test_k_class_assemble_result_can_keep_full_accumulators_on_host():
    result = _assemble_result(
        class_log_evidence=np.asarray([[0.0], [-0.2]], dtype=np.float64),
        new_means=None,
        Ft_y=[jnp.asarray([1.0, 2.0]), jnp.asarray([3.0, 4.0])],
        Ft_ctf=[jnp.asarray([5.0, 6.0]), jnp.asarray([7.0, 8.0])],
        per_class_hard_assignments=np.asarray([[3], [7]], dtype=np.int32),
        per_class_stats=(
            _stats([0.0], [-5.0], [0.01]),
            _stats([-0.2], [-1.0], [0.20]),
        ),
        noise_stats=None,
        host_accumulators=True,
    )

    assert isinstance(result.Ft_y, np.ndarray)
    assert isinstance(result.Ft_ctf, np.ndarray)
    np.testing.assert_allclose(result.Ft_y, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(result.Ft_ctf, np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))


def test_k_class_combined_accumulator_skips_empty_half_allocation():
    half = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    assert _combine_optional_half_accumulators(half, None, label="Ft_y") is half
    assert _combine_optional_half_accumulators(None, half, label="Ft_y") is half
    np.testing.assert_allclose(
        np.asarray(_combine_optional_half_accumulators(half, half, label="Ft_y")),
        np.asarray([2.0, 4.0], dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="Ft_y accumulators"):
        _combine_optional_half_accumulators(None, None, label="Ft_y")


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


def test_adaptive_k_class_firstiter_override_redecodes_best_pose_details(monkeypatch):
    score_calls = []
    probe_calls = []
    dense_calls = []

    class TinyDataset:
        n_images = 2

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
        score_calls.append(kwargs)
        n_images = TinyDataset.n_images
        n_rot = int(np.asarray(rotations_arg).shape[0])
        class_index = len(score_calls) - 1
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

    def fake_run_dense_k_class_em(
        _dataset,
        means,
        _mean_variance,
        _noise_variance,
        rotations_arg,
        _translations,
        _disc_type,
        **kwargs,
    ):
        dense_calls.append(kwargs)
        n_images = TinyDataset.n_images
        n_classes = int(np.asarray(means).shape[0])
        n_rot = int(np.asarray(rotations_arg).shape[0])
        hard = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
        scores = (np.asarray([5.0, 1.0], dtype=np.float32), np.asarray([1.0, 5.0], dtype=np.float32))
        best_rots, best_trans, best_rot_ids = zip(*(pose_details(row) for row in hard), strict=True)
        return _assemble_result(
            class_log_evidence=np.zeros((n_classes, n_images), dtype=np.float64),
            new_means=[jnp.zeros(4, dtype=jnp.complex64) for _ in range(n_classes)],
            Ft_y=[jnp.zeros(4, dtype=jnp.complex64) for _ in range(n_classes)],
            Ft_ctf=[jnp.zeros(4, dtype=jnp.float32) for _ in range(n_classes)],
            per_class_hard_assignments=hard,
            per_class_stats=tuple(
                make_relion_stats(
                    log_evidence_per_image=np.zeros(n_images, dtype=np.float32),
                    best_log_score_per_image=scores[class_idx],
                    max_posterior_per_image=np.ones(n_images, dtype=np.float32),
                    rotation_posterior_sums=np.zeros(n_rot, dtype=np.float32),
                )
                for class_idx in range(n_classes)
            ),
            noise_stats=None,
            per_class_best_pose_rotations=best_rots,
            per_class_best_pose_translations=best_trans,
            per_class_best_pose_rotation_ids=best_rot_ids,
        )

    monkeypatch.setattr(k_class_module, "run_em", fake_run_em)
    monkeypatch.setattr(k_class_module, "run_dense_k_class_em", fake_run_dense_k_class_em)
    def fake_joint_probe(*args, **kwargs):
        probe_calls.append((args, kwargs))
        return _firstiter_probe_result(
            [1, 0],
            per_class_hard=np.asarray([[0, 0], [1, 1]], dtype=np.int32),
            n_rot=1,
        )

    monkeypatch.setattr(k_class_module, "_run_dense_k_class_joint_firstiter_score_probe", fake_joint_probe)

    result = run_dense_k_class_em_adaptive(
        TinyDataset(),
        jnp.zeros((2, 4), dtype=jnp.complex64),
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

    assert len(probe_calls) == 1
    assert len(score_calls) == 0
    assert len(dense_calls) == 1
    assert dense_calls[0]["return_best_pose_details"]
    assert dense_calls[0]["sparse_pass2"] is False
    np.testing.assert_array_equal(np.asarray(result.class_assignments), np.asarray([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(result.pose_assignments), np.asarray([2, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(result.best_pose_rotation_ids), np.asarray([1, 0], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(result.best_pose_translations), fine_translations[[0, 1]])


def test_firstiter_score_probe_uses_joint_significance(monkeypatch):
    from recovar.em.dense_single_volume.helpers import significance as significance_module

    calls = []

    class TinyDataset:
        n_units = 3

    def fake_compute_significance(*args, **kwargs):
        calls.append(kwargs)
        return (
            None,
            None,
            None,
            np.asarray([1, 0, 1], dtype=np.int32),
            None,
            {
                "class_log_evidence_per_image": np.asarray(
                    [[0.0, 2.0, 1.0], [1.0, 0.0, 3.0]],
                    dtype=np.float64,
                ),
                "class_hard_assignments": np.asarray([[4, 5, 6], [7, 8, 9]], dtype=np.int32),
                "class_best_log_score_per_image": np.asarray(
                    [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
                    dtype=np.float32,
                ),
                "class_assignments": np.asarray([1, 0, 1], dtype=np.int32),
            },
        )

    def fail_run_em(*_args, **_kwargs):
        raise AssertionError("firstiter K-class coarse probe should not call run_em per class")

    monkeypatch.setattr(significance_module, "_compute_k_class_significance_batched", fake_compute_significance)
    monkeypatch.setattr(k_class_module, "run_em", fail_run_em)

    result = k_class_module._run_dense_k_class_score_probe(
        TinyDataset(),
        jnp.zeros((2, 4), dtype=jnp.complex64),
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        np.zeros((5, 3, 3), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
        "linear_interp",
        class_log_priors=np.log(np.asarray([0.4, 0.6], dtype=np.float64)),
        relion_firstiter_score_mode="normalized_cc",
        relion_firstiter_winner_take_all=True,
        current_size=26,
        image_batch_size=7,
        rotation_block_size=5,
    )

    assert len(calls) == 1
    assert calls[0]["score_mode"] == "normalized_cc"
    assert calls[0]["collect_significance"] is False
    assert calls[0]["return_class_best"] is True
    assert calls[0]["current_size"] == 26
    np.testing.assert_array_equal(result.class_assignments, np.asarray([1, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(result.per_class_hard_assignments, np.asarray([[4, 5, 6], [7, 8, 9]], dtype=np.int32))


def test_adaptive_k_class_firstiter_uses_coarse_current_size_for_probe(monkeypatch):
    score_calls = []
    probe_calls = []
    dense_calls = []

    class TinyDataset:
        n_images = 2
        image_shape = (256, 256)

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
        n_images = TinyDataset.n_images
        n_rot = int(np.asarray(rotations).shape[0])
        score_calls.append(kwargs)
        stats = make_relion_stats(
            log_evidence_per_image=np.zeros(n_images, dtype=np.float32),
            best_log_score_per_image=np.zeros(n_images, dtype=np.float32),
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

    def fake_run_dense_k_class_em(
        _dataset,
        means,
        _mean_variance,
        _noise_variance,
        rotations,
        _translations,
        _disc_type,
        **kwargs,
    ):
        dense_calls.append(kwargs)
        n_images = TinyDataset.n_images
        n_classes = int(np.asarray(means).shape[0])
        n_rot = int(np.asarray(rotations).shape[0])
        stats = tuple(
            make_relion_stats(
                log_evidence_per_image=np.zeros(n_images, dtype=np.float32),
                best_log_score_per_image=np.zeros(n_images, dtype=np.float32),
                max_posterior_per_image=np.ones(n_images, dtype=np.float32),
                rotation_posterior_sums=np.zeros(n_rot, dtype=np.float32),
            )
            for _ in range(n_classes)
        )
        return _assemble_result(
            class_log_evidence=np.zeros((n_classes, n_images), dtype=np.float64),
            new_means=[jnp.zeros(4, dtype=jnp.complex64) for _ in range(n_classes)],
            Ft_y=[jnp.zeros(4, dtype=jnp.complex64) for _ in range(n_classes)],
            Ft_ctf=[jnp.zeros(4, dtype=jnp.float32) for _ in range(n_classes)],
            per_class_hard_assignments=np.zeros((n_classes, n_images), dtype=np.int32),
            per_class_stats=stats,
            noise_stats=None,
        )

    monkeypatch.setattr(k_class_module, "run_em", fake_run_em)
    monkeypatch.setattr(k_class_module, "run_dense_k_class_em", fake_run_dense_k_class_em)
    def fake_joint_probe(*_args, **kwargs):
        probe_calls.append(kwargs["engine_kwargs"])
        return _firstiter_probe_result(
            [0, 0],
            per_class_hard=np.zeros((2, 2), dtype=np.int32),
            n_rot=2,
        )

    monkeypatch.setattr(k_class_module, "_run_dense_k_class_joint_firstiter_score_probe", fake_joint_probe)

    run_dense_k_class_em_adaptive(
        TinyDataset(),
        jnp.zeros((2, 4), dtype=jnp.complex64),
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        np.zeros((2, 3, 3), dtype=np.float32),
        np.zeros((29, 2), dtype=np.float32),
        np.zeros((4, 3, 3), dtype=np.float32),
        np.zeros((116, 2), dtype=np.float32),
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.arange(116, dtype=np.int64) % 29,
        "linear_interp",
        coarse_current_size=26,
        fine_current_size=56,
        current_size=56,
        firstiter_cc_pass2_only_best_coarse=True,
    )

    assert len(probe_calls) == 1
    assert len(score_calls) == 0
    assert len(dense_calls) == 1
    assert probe_calls[0]["current_size"] == 26
    assert dense_calls[0]["current_size"] == 56
    assert dense_calls[0]["sparse_pass2"] is False


def test_adaptive_k_class_firstiter_fine_pass_uses_global_winner_subsets(monkeypatch):
    calls = []
    probe_calls = []

    class TinyDataset:
        def __init__(self, indices=None):
            self.indices = np.arange(4, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
            self.n_units = int(self.indices.size)
            self.n_images = int(self.indices.size)

        def subset(self, indices):
            return TinyDataset(self.indices[np.asarray(indices, dtype=np.int64)])

    def fake_run_em(
        dataset,
        mean,
        _mean_variance,
        _noise_variance,
        rotations,
        _translations,
        _disc_type,
        **kwargs,
    ):
        class_index = int(np.real(np.asarray(mean)[0]))
        is_fine = int(np.asarray(rotations).shape[0]) == 2
        calls.append((is_fine, tuple(dataset.indices.tolist()), kwargs))
        n_images = int(dataset.n_units)
        if is_fine:
            expected_corr = np.arange(4, dtype=np.float32)[np.asarray(dataset.indices, dtype=np.int64)]
            np.testing.assert_array_equal(np.asarray(kwargs["image_corrections"]), expected_corr)
            hard = np.arange(n_images, dtype=np.int32) % 2
            best = np.full(n_images, 10.0 + class_index, dtype=np.float32)
        elif class_index == 0:
            hard = np.zeros(n_images, dtype=np.int32)
            best = np.asarray([10.0, 1.0, 1.0, 10.0], dtype=np.float32)
        else:
            hard = np.zeros(n_images, dtype=np.int32)
            best = np.asarray([1.0, 10.0, 10.0, 1.0], dtype=np.float32)
        stats = make_relion_stats(
            log_evidence_per_image=np.zeros(n_images, dtype=np.float32),
            best_log_score_per_image=best,
            max_posterior_per_image=np.ones(n_images, dtype=np.float32),
            rotation_posterior_sums=np.zeros(int(np.asarray(rotations).shape[0]), dtype=np.float32),
        )
        return (
            jnp.zeros_like(mean),
            hard,
            jnp.ones_like(mean) * (class_index + 1),
            jnp.ones_like(jnp.real(mean)) * (class_index + 2),
            stats,
        )

    monkeypatch.setattr(k_class_module, "run_em", fake_run_em)
    def fake_joint_probe(*args, **kwargs):
        probe_calls.append((args, kwargs))
        return _firstiter_probe_result(
            [0, 1, 1, 0],
            per_class_hard=np.zeros((2, 4), dtype=np.int32),
            n_rot=1,
        )

    monkeypatch.setattr(k_class_module, "_run_dense_k_class_joint_firstiter_score_probe", fake_joint_probe)

    result = run_dense_k_class_em_adaptive(
        TinyDataset(),
        jnp.asarray([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=jnp.complex64),
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        np.zeros((1, 3, 3), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((2, 3, 3), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
        "linear_interp",
        firstiter_cc_pass2_only_best_coarse=True,
        image_corrections=np.arange(4, dtype=np.float32),
    )

    fine_calls = [call for call in calls if call[0]]
    assert len(probe_calls) == 1
    assert [call[1] for call in fine_calls] == [(0, 3), (1, 2)]
    np.testing.assert_array_equal(np.asarray(result.class_assignments), np.asarray([0, 1, 1, 0], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(result.class_posterior_sums), np.asarray([2.0, 2.0], dtype=np.float32))


def test_adaptive_k_class_firstiter_sparse_fine_pass_uses_global_winner_subsets(monkeypatch):
    from recovar.em.dense_single_volume.helpers import oversampling as oversampling_module
    from recovar.em.sampling import rotation_grid_size

    score_calls = []
    probe_calls = []
    sparse_calls = []
    n_coarse_rot = rotation_grid_size(0)
    n_fine_rot = rotation_grid_size(1)
    n_fine_trans = 2

    class TinyDataset:
        def __init__(self, indices=None):
            self.indices = np.arange(4, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
            self.n_units = int(self.indices.size)
            self.n_images = int(self.indices.size)

        def subset(self, indices):
            return TinyDataset(self.indices[np.asarray(indices, dtype=np.int64)])

    def fake_run_em(
        dataset,
        mean,
        _mean_variance,
        _noise_variance,
        rotations,
        translations,
        _disc_type,
        **kwargs,
    ):
        class_index = int(np.real(np.asarray(mean)[0]))
        score_calls.append((class_index, tuple(dataset.indices.tolist()), kwargs))
        n_images = int(dataset.n_units)
        hard = np.arange(n_images, dtype=np.int32) % (int(np.asarray(rotations).shape[0]) * int(np.asarray(translations).shape[0]))
        best = (
            np.asarray([10.0, 1.0, 1.0, 10.0], dtype=np.float32),
            np.asarray([1.0, 10.0, 10.0, 1.0], dtype=np.float32),
        )[class_index]
        stats = make_relion_stats(
            log_evidence_per_image=np.zeros(n_images, dtype=np.float32),
            best_log_score_per_image=best,
            max_posterior_per_image=np.ones(n_images, dtype=np.float32),
            rotation_posterior_sums=np.zeros(int(np.asarray(rotations).shape[0]), dtype=np.float32),
        )
        return (
            jnp.zeros_like(mean),
            hard,
            jnp.zeros_like(mean),
            jnp.zeros_like(jnp.real(mean)),
            stats,
        )

    def fake_compute_pass2_stats_sparse(
        dataset,
        volume,
        _mean_variance,
        _noise_variance,
        _translations,
        significant_sample_indices,
        **kwargs,
    ):
        class_index = int(np.real(np.asarray(volume)[0]))
        sparse_calls.append((class_index, tuple(dataset.indices.tolist()), significant_sample_indices, kwargs))
        np.testing.assert_array_equal(
            np.asarray(kwargs["image_corrections"]),
            np.arange(4, dtype=np.float32)[np.asarray(dataset.indices, dtype=np.int64)],
        )
        assert kwargs["relion_firstiter_score_mode"] == "normalized_cc"
        assert kwargs["relion_firstiter_winner_take_all"] is True
        n_images = int(dataset.n_units)
        hard = np.arange(n_images, dtype=np.int32) % n_fine_trans
        best_rot_ids = np.full(n_images, class_index, dtype=np.int32)
        stats = make_relion_stats(
            log_evidence_per_image=np.zeros(n_images, dtype=np.float32),
            best_log_score_per_image=np.full(n_images, 10.0 + class_index, dtype=np.float32),
            max_posterior_per_image=np.ones(n_images, dtype=np.float32),
            rotation_posterior_sums=np.zeros(n_coarse_rot, dtype=np.float32),
        )
        return (
            jnp.ones_like(volume) * (class_index + 1),
            jnp.ones_like(jnp.real(volume)) * (class_index + 2),
            hard,
            np.repeat(np.eye(3, dtype=np.float32)[None], n_images, axis=0),
            np.zeros((n_images, 2), dtype=np.float32),
            best_rot_ids,
            stats,
        )

    monkeypatch.setattr(k_class_module, "run_em", fake_run_em)
    monkeypatch.setattr(oversampling_module, "compute_pass2_stats_sparse", fake_compute_pass2_stats_sparse)
    def fake_joint_probe(*args, **kwargs):
        probe_calls.append((args, kwargs))
        return _firstiter_probe_result(
            [0, 1, 1, 0],
            per_class_hard=np.zeros((2, 4), dtype=np.int32),
            n_rot=n_coarse_rot,
        )

    monkeypatch.setattr(k_class_module, "_run_dense_k_class_joint_firstiter_score_probe", fake_joint_probe)

    result = run_dense_k_class_em_adaptive(
        TinyDataset(),
        jnp.asarray([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=jnp.complex64),
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        np.zeros((n_coarse_rot, 3, 3), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((n_fine_rot, 3, 3), dtype=np.float32),
        np.zeros((n_fine_trans, 2), dtype=np.float32),
        np.arange(n_fine_rot, dtype=np.int64) % n_coarse_rot,
        np.zeros(n_fine_trans, dtype=np.int64),
        "linear_interp",
        firstiter_cc_pass2_only_best_coarse=True,
        sparse_pass2=True,
        image_corrections=np.arange(4, dtype=np.float32),
    )

    assert [call[1] for call in sparse_calls] == [(0, 3), (1, 2)]
    assert len(probe_calls) == 1
    assert len(score_calls) == 0
    np.testing.assert_array_equal(np.asarray(result.class_assignments), np.asarray([0, 1, 1, 0], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(result.class_posterior_sums), np.asarray([2.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(result.pose_assignments), np.asarray([0, 2, 3, 1], dtype=np.int32))


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


def test_sparse_k_class_adaptive_mstep_uses_score_space_log_z(monkeypatch):
    """Sparse K-class pass-2 normalizes scores, not evidence plus image offset."""

    from recovar.em.dense_single_volume.helpers import oversampling as oversampling_module
    from recovar.em.sampling import rotation_grid_size

    calls = []
    n_coarse_rot = rotation_grid_size(1)
    n_fine_rot = rotation_grid_size(2)

    class TinyDataset:
        n_units = 2

    probe_log_evidence = [
        np.asarray([1000.0, 2000.0], dtype=np.float64),
        np.asarray([1005.0, 1997.0], dtype=np.float64),
    ]
    probe_score_log_z = [
        np.asarray([1.0, 2.0], dtype=np.float64),
        np.asarray([3.0, 0.0], dtype=np.float64),
    ]

    def fake_compute_pass2_stats_sparse(
        _dataset,
        volume,
        _mean_variance,
        _noise_variance,
        _translations,
        _significant_sample_indices,
        **kwargs,
    ):
        calls.append(kwargs)
        class_index = (len(calls) - 1) % 2
        n_images = TinyDataset.n_units
        if kwargs.get("return_score_log_z_only"):
            return probe_log_evidence[class_index], probe_score_log_z[class_index]
        stats = make_relion_stats(
            log_evidence_per_image=probe_log_evidence[class_index],
            best_log_score_per_image=np.full(n_images, float(class_index), dtype=np.float64),
            max_posterior_per_image=np.full(n_images, 0.5, dtype=np.float32),
            rotation_posterior_sums=np.zeros(n_coarse_rot, dtype=np.float32),
        )
        common = (
            jnp.zeros_like(volume),
            jnp.zeros_like(volume),
            np.zeros(n_images, dtype=np.int32),
            np.repeat(np.eye(3, dtype=np.float32)[None], n_images, axis=0),
            np.zeros((n_images, 2), dtype=np.float32),
            np.zeros(n_images, dtype=np.int32),
            stats,
        )
        if kwargs.get("return_score_log_z"):
            return common + (probe_score_log_z[class_index],)
        return common

    monkeypatch.setattr(oversampling_module, "compute_pass2_stats_sparse", fake_compute_pass2_stats_sparse)

    result = _run_sparse_k_class_adaptive_pass2(
        TinyDataset(),
        jnp.zeros((2, 4), dtype=jnp.complex64),
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(4, dtype=jnp.float32),
        np.repeat(np.eye(3, dtype=np.float32)[None], n_coarse_rot, axis=0),
        np.zeros((1, 2), dtype=np.float32),
        np.repeat(np.eye(3, dtype=np.float32)[None], n_fine_rot, axis=0),
        np.repeat(np.arange(n_coarse_rot, dtype=np.int64), n_fine_rot // n_coarse_rot),
        np.zeros((2, 2), dtype=np.float32),
        np.asarray([0, 0], dtype=np.int64),
        [[np.asarray([0], dtype=np.int32)] * TinyDataset.n_units for _ in range(2)],
        "linear_interp",
        class_log_priors=np.log(np.asarray([0.5, 0.5], dtype=np.float64)),
        accumulate_noise=False,
        return_best_pose_details=False,
        oversampling_order=1,
        random_perturbation=0.0,
        engine_kwargs={"relion_half_volume_mstep": True},
    )

    assert len(calls) == 3
    assert calls[0].get("return_score_log_z_only")
    assert calls[1].get("return_score_log_z")
    np.testing.assert_allclose(calls[1]["normalization_other_score_log_z"], probe_score_log_z[0])
    assert all(call.get("relion_half_volume_mstep") is True for call in calls)
    expected_score_log_z = np.logaddexp(probe_score_log_z[0], probe_score_log_z[1])
    np.testing.assert_allclose(calls[2]["normalization_log_z"], expected_score_log_z)
    evidence_log_z = np.logaddexp(probe_log_evidence[0], probe_log_evidence[1])
    assert not np.allclose(expected_score_log_z, evidence_log_z)
    assert isinstance(result.Ft_y, np.ndarray)
    assert isinstance(result.Ft_ctf, np.ndarray)


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


def test_class3d_replay_loads_shared_model_direction_prior(tmp_path, monkeypatch):
    """Class3D replay uses run_itNNN_model.star when half-model files are absent."""

    from recovar.em.dense_single_volume import iteration_loop
    from recovar.em.dense_single_volume.relion_replay import (
        _RelionHalfInputState,
        apply_iter_replay_overrides,
    )

    (tmp_path / "run_it001_model.star").touch()
    (tmp_path / "run_it002_model.star").touch()

    raw_prior = np.zeros((2, 12), dtype=np.float32)
    raw_prior[0, :2] = [0.20, 0.10]
    raw_prior[1, :2] = [0.15, 0.55]

    monkeypatch.setattr(
        iteration_loop,
        "read_relion_sampling_metadata",
        lambda _path: {
            "healpix_order": 0,
            "psi_step": 7.5,
            "offset_range": 4.0,
            "offset_step": 2.0,
        },
    )
    monkeypatch.setattr(
        iteration_loop,
        "read_relion_model_metadata",
        lambda _path: {"current_image_size": 8},
    )
    monkeypatch.setattr(iteration_loop, "get_translation_grid", lambda _range, _step: np.zeros((1, 2), dtype=np.float32))
    calls = []

    def fake_read_priors(path, n_classes):
        calls.append(str(path))
        assert n_classes == 2
        return raw_prior

    monkeypatch.setattr(iteration_loop, "read_relion_direction_priors", fake_read_priors)

    state = SimpleNamespace(
        healpix_order=0,
        max_healpix_order=4,
        auto_local_healpix_order=99,
        do_local_search=False,
        sigma_rot=0.0,
        sigma_psi=0.0,
        translation_range=0.0,
        translation_step=1.0,
    )
    half_inputs = _RelionHalfInputState.from_initial_values(
        previous_best_translations=None,
        previous_best_rotation_eulers=None,
        image_corrections=None,
        scale_corrections=None,
    )
    class_priors = [None, None]
    class_orders = [None, None]

    result = apply_iter_replay_overrides(
        iter_replay_override=None,
        perturb_replay_relion_dir=str(tmp_path),
        init_relion_iteration=0,
        iteration=1,
        state=state,
        cs=4,
        cryo=SimpleNamespace(voxel_size=1.0, image_shape=(8, 8)),
        k_class_enabled=True,
        n_classes=2,
        relion_half_inputs=half_inputs,
        previous_best_rotations=[None, None],
        noise_variance_per_half=[None, None],
        noise_variance=None,
        previous_noise_radial_per_half=[None, None],
        previous_noise_radial=None,
        current_sigma_offset_angstrom=1.0,
        class_direction_prior_per_half=class_priors,
        class_direction_prior_order_per_half=class_orders,
        global_direction_prior_per_half=[None, None],
        global_direction_prior_order_per_half=[None, None],
    )

    assert calls == [str(tmp_path / "run_it001_model.star"), str(tmp_path / "run_it001_model.star")]
    np.testing.assert_allclose(class_priors[0].sum(axis=1), np.ones(2), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(class_priors[1], class_priors[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.class_weights, raw_prior.sum(axis=1) / raw_prior.sum(), rtol=1e-6, atol=1e-6)


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
