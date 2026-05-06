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
    _assemble_result,
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


def test_adaptive_k_class_firstiter_override_redecodes_best_pose_details(monkeypatch):
    calls = []

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
        calls.append(kwargs)
        n_images = TinyDataset.n_images
        n_classes = int(np.asarray(means).shape[0])
        n_rot = int(np.asarray(rotations_arg).shape[0])
        is_coarse = len(calls) == 1
        if is_coarse:
            hard = np.asarray([[0, 0], [1, 1]], dtype=np.int32)
            scores = (np.asarray([1.0, 5.0], dtype=np.float32), np.asarray([5.0, 1.0], dtype=np.float32))
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
            )

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

    monkeypatch.setattr(k_class_module, "run_dense_k_class_em", fake_run_dense_k_class_em)

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

    assert calls[1]["return_best_pose_details"]
    np.testing.assert_array_equal(np.asarray(result.class_assignments), np.asarray([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(result.pose_assignments), np.asarray([2, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(result.best_pose_rotation_ids), np.asarray([1, 0], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(result.best_pose_translations), fine_translations[[0, 1]])


def test_adaptive_k_class_firstiter_uses_coarse_current_size_for_probe(monkeypatch):
    calls = []

    class TinyDataset:
        n_images = 2

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
        calls.append(kwargs)
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

    monkeypatch.setattr(k_class_module, "run_dense_k_class_em", fake_run_dense_k_class_em)

    run_dense_k_class_em_adaptive(
        TinyDataset(),
        jnp.zeros((2, 4), dtype=jnp.complex64),
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
    )

    assert len(calls) == 2
    assert calls[0]["current_size"] == 26
    assert calls[1]["current_size"] == 56


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
