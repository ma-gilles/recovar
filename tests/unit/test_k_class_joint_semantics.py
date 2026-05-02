import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.em.dense_single_volume.k_class as k_class_module
from recovar.em.dense_single_volume.helpers.orientation_priors import (
    class_weights_from_direction_prior,
    normalize_class_direction_prior_per_half,
)
from recovar.em.dense_single_volume.helpers.types import make_relion_stats
from recovar.em.dense_single_volume.k_class import _assemble_result, run_dense_k_class_em, run_local_k_class_em
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
