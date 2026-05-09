"""InitialModel dense K-class E-step adapter tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume.local_layout import LocalHypothesisLayout
from recovar.em.initial_model import initialise_denovo_state
from recovar.em.initial_model.dense_adapter import (
    DenseInitialModelEstepConfig,
    _estep_meta,
    _initial_model_pass2_layout,
    _relion_projector_to_dense_volume,
    _resolve_sparse_pass1_current_size,
    class_log_priors_from_state,
    reference_to_dense_means,
    reference_to_relion_projector_dense_means,
    run_dense_initial_model_estep,
    split_pseudo_halfset_particle_ids,
)

pytestmark = pytest.mark.unit


class _Dataset:
    n_images = 4

    def subset(self, image_indices):
        n_images = int(np.asarray(image_indices).size)
        return SimpleNamespace(n_images=n_images, n_units=n_images)


def _fake_result(n_classes: int, n: int, *, n_images: int = 2, n_groups: int = 2):
    Ft_y = [np.full(n**3, k + 1, dtype=np.complex64) for k in range(n_classes)]
    Ft_ctf = [np.full(n**3, (k + 1) * 2, dtype=np.float32) for k in range(n_classes)]
    per_class_stats = tuple(
        SimpleNamespace(rotation_posterior_sums=np.full(3, k + 1, dtype=np.float32)) for k in range(n_classes)
    )
    return SimpleNamespace(
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        grouped_Ft_y=np.broadcast_to(np.asarray(Ft_y)[None, :, :], (n_groups, n_classes, n**3)).copy(),
        grouped_Ft_ctf=np.broadcast_to(np.asarray(Ft_ctf)[None, :, :], (n_groups, n_classes, n**3)).copy(),
        class_responsibilities=np.full((n_classes, n_images), 1.0 / n_classes, dtype=np.float32),
        class_posterior_sums=np.arange(n_classes, dtype=np.float32),
        class_assignments=np.zeros(n_images, dtype=np.int32),
        pose_assignments=np.arange(n_images, dtype=np.int32),
        best_pose_rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (n_images, 3, 3)).copy(),
        best_pose_translations=np.arange(n_images * 2, dtype=np.float32).reshape(n_images, 2),
        best_pose_rotation_ids=np.arange(n_images, dtype=np.int32),
        stats=SimpleNamespace(max_posterior_per_image=np.linspace(0.25, 0.75, n_images, dtype=np.float32)),
        per_class_stats=per_class_stats,
    )


def _fake_noise_stats(offset: float, sumw: float, wsum_noise, img_power):
    return SimpleNamespace(
        wsum_sigma2_offset=float(offset),
        sumw=float(sumw),
        wsum_sigma2_noise=np.asarray(wsum_noise, dtype=np.float32),
        wsum_img_power=np.asarray(img_power, dtype=np.float32),
    )


def _fake_result_with_profile(n_classes: int, n: int, *, n_images: int = 2, n_groups: int = 2):
    result = _fake_result(n_classes, n, n_images=n_images, n_groups=n_groups)
    result.profile_summary = {"em_time_s": 1.25, "batches": 1}
    return result


def test_split_pseudo_halfset_particle_ids_uses_particle_id_parity():
    h0, h1 = split_pseudo_halfset_particle_ids(
        5,
        micrograph_names=np.asarray(["b", "a", "b", "a", "c"]),
    )
    np.testing.assert_array_equal(h0, np.asarray([0, 2, 4]))
    np.testing.assert_array_equal(h1, np.asarray([1, 3]))


def test_class_log_priors_from_state_normalizes_weights():
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=2, nr_iter=1, n_directions=4)
    state.pdf_class = np.asarray([2.0, 1.0])
    np.testing.assert_allclose(class_log_priors_from_state(state), np.log([2.0 / 3.0, 1.0 / 3.0]))


def test_class_log_priors_from_state_allows_inactive_class():
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=2, nr_iter=1, n_directions=4)
    state.pdf_class = np.asarray([1.0, 0.0])
    out = class_log_priors_from_state(state)
    assert out[0] == 0.0
    assert out[1] < -1.0e20


def test_dense_initial_model_estep_runs_separate_k_class_calls_for_pseudo_halfsets(monkeypatch):
    calls = []

    def fake_run_dense_k_class_em(
        dataset, means, mean_variance, noise_variance, rotations, translations, disc_type, **kwargs
    ):
        calls.append(
            {
                "means_shape": np.asarray(means).shape,
                "image_indices": np.asarray(kwargs["image_indices"]).copy(),
                "has_reconstruction_group_ids": "reconstruction_group_ids" in kwargs,
                "has_reconstruction_group_count": "reconstruction_group_count" in kwargs,
                "class_log_priors": np.asarray(kwargs["class_log_priors"]).copy(),
                "current_size": kwargs["current_size"],
            }
        )
        return _fake_result(n_classes=2, n=8, n_images=int(np.asarray(kwargs["image_indices"]).size), n_groups=1)

    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.run_dense_k_class_em",
        fake_run_dense_k_class_em,
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=2, nr_iter=1, n_directions=4)
    state.current_size = 8
    state.pdf_class = np.asarray([0.75, 0.25])
    config = DenseInitialModelEstepConfig(
        means=np.zeros((2, 8**3), dtype=np.complex64),
        mean_variance=np.ones((2, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_bpref_frame=False,
    )

    result = run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([0, 1, 2, 3]),
        halfset_ids=np.asarray([0, 1, 0, 1], dtype=np.int8),
    )

    assert len(calls) == 2
    assert calls[0]["means_shape"] == (2, 8**3)
    assert calls[1]["means_shape"] == (2, 8**3)
    np.testing.assert_array_equal(calls[0]["image_indices"], [0, 2])
    np.testing.assert_array_equal(calls[1]["image_indices"], [1, 3])
    assert calls[0]["has_reconstruction_group_ids"] is False
    assert calls[1]["has_reconstruction_group_ids"] is False
    assert calls[0]["has_reconstruction_group_count"] is False
    assert calls[1]["has_reconstruction_group_count"] is False
    np.testing.assert_allclose(calls[0]["class_log_priors"], np.log([0.75, 0.25]))
    np.testing.assert_allclose(calls[1]["class_log_priors"], np.log([0.75, 0.25]))
    assert calls[0]["current_size"] == 8
    assert calls[1]["current_size"] == 8

    assert len(result.accumulators) == 4
    assert [(a.halfset_idx, a.class_idx) for a in result.accumulators] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for accum in result.accumulators:
        assert accum.data.shape == (8, 8, 5)
        assert accum.weight.shape == (8, 8, 5)
    np.testing.assert_allclose(result.accumulators[0].data, 1.0)
    np.testing.assert_allclose(result.accumulators[1].weight, 4.0)
    assert result.meta["halfset_ids"] == (0, 1)
    assert "fused_pseudo_halfsets" not in result.meta
    np.testing.assert_allclose(result.meta["class_posterior_sums"], [0.0, 2.0])
    np.testing.assert_allclose(
        result.meta["class_direction_posterior_sums"],
        np.asarray([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]),
    )
    np.testing.assert_array_equal(result.meta["selected_particle_ids"], [0, 2, 1, 3])
    np.testing.assert_array_equal(result.meta["pose_assignments"], [0, 1, 0, 1])
    np.testing.assert_array_equal(result.meta["class_assignments"], [0, 0, 0, 0])
    np.testing.assert_array_equal(result.meta["best_pose_rotation_ids"], [0, 1, 0, 1])
    np.testing.assert_allclose(
        result.meta["best_pose_translations"],
        np.asarray([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=np.float32),
    )
    assert result.meta["best_pose_rotations"].shape == (4, 3, 3)
    np.testing.assert_allclose(
        result.meta["max_posterior_per_image"],
        np.asarray([0.25, 0.75, 0.25, 0.75], dtype=np.float32),
    )


def test_estep_meta_aggregates_noise_stats_for_model_updates():
    halfset_results = {
        0: SimpleNamespace(
            class_posterior_sums=np.asarray([1.0, 2.0], dtype=np.float32),
            aggregate_noise_stats=_fake_noise_stats(10.0, 3.0, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        ),
        1: SimpleNamespace(
            class_posterior_sums=np.asarray([3.0, 4.0], dtype=np.float32),
            aggregate_noise_stats=_fake_noise_stats(20.0, 7.0, [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]),
        ),
    }

    meta = _estep_meta(halfset_results)

    assert meta["wsum_sigma2_offset"] == pytest.approx(30.0)
    assert meta["sigma2_offset_sumw"] == pytest.approx(10.0)
    assert meta["noise_sumw"] == pytest.approx(10.0)
    np.testing.assert_allclose(meta["wsum_sigma2_noise"], [8.0, 10.0, 12.0])
    np.testing.assert_allclose(meta["wsum_img_power"], [14.0, 16.0, 18.0])
    np.testing.assert_allclose(meta["halfset_0_wsum_sigma2_noise"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(meta["halfset_1_wsum_img_power"], [10.0, 11.0, 12.0])


def test_dense_initial_model_estep_slices_full_translation_prior_for_pseudo_halfsets(monkeypatch):
    calls = []

    def fake_run_dense_k_class_em(*args, **kwargs):
        calls.append(np.asarray(kwargs["translation_log_prior"]).copy())
        return _fake_result(n_classes=1, n=8, n_images=int(np.asarray(kwargs["image_indices"]).size), n_groups=1)

    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.run_dense_k_class_em",
        fake_run_dense_k_class_em,
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4)
    full_prior = np.arange(8, dtype=np.float32).reshape(4, 2)
    config = DenseInitialModelEstepConfig(
        means=np.zeros((1, 8**3), dtype=np.complex64),
        mean_variance=np.ones((1, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((2, 2), dtype=np.float32),
        relion_bpref_frame=False,
        engine_kwargs={"translation_log_prior": full_prior},
    )

    run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([0, 1, 2, 3]),
        halfset_ids=np.asarray([0, 1, 0, 1], dtype=np.int8),
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], full_prior[[0, 2]])
    np.testing.assert_array_equal(calls[1], full_prior[[1, 3]])


def test_dense_initial_model_estep_meta_includes_optional_profiles(monkeypatch):
    def fake_run_dense_k_class_em(*args, **kwargs):
        assert kwargs["return_profile"] is True
        return _fake_result_with_profile(
            n_classes=1,
            n=8,
            n_images=int(np.asarray(kwargs["image_indices"]).size),
        )

    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.run_dense_k_class_em",
        fake_run_dense_k_class_em,
    )
    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=1,
        nr_iter=1,
        n_directions=4,
        pseudo_halfsets=False,
    )
    config = DenseInitialModelEstepConfig(
        means=np.zeros((1, 8**3), dtype=np.complex64),
        mean_variance=np.ones((1, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_bpref_frame=False,
        engine_kwargs={"return_profile": True},
    )

    result = run_dense_initial_model_estep(_Dataset(), state, config)

    assert result.meta["halfset_0_profile_summary"] == {"em_time_s": 1.25, "batches": 1}
    np.testing.assert_allclose(result.meta["class_posterior_sums"], [0.0])
    np.testing.assert_allclose(result.meta["class_direction_posterior_sums"], [[1.0, 1.0, 1.0]])
    np.testing.assert_array_equal(result.meta["selected_particle_ids"], [0, 1, 2, 3])
    np.testing.assert_array_equal(result.meta["class_assignments"], [0, 0, 0, 0])
    np.testing.assert_allclose(
        result.meta["max_posterior_per_image"],
        np.linspace(0.25, 0.75, 4, dtype=np.float32),
    )


def test_dense_initial_model_estep_pseudo_halfset_meta_includes_per_halfset_profiles(monkeypatch):
    def fake_run_dense_k_class_em(*args, **kwargs):
        assert kwargs["return_profile"] is True
        return _fake_result_with_profile(
            n_classes=1,
            n=8,
            n_images=int(np.asarray(kwargs["image_indices"]).size),
            n_groups=1,
        )

    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.run_dense_k_class_em",
        fake_run_dense_k_class_em,
    )
    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=1,
        nr_iter=1,
        n_directions=4,
        pseudo_halfsets=True,
    )
    config = DenseInitialModelEstepConfig(
        means=np.zeros((1, 8**3), dtype=np.complex64),
        mean_variance=np.ones((1, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_bpref_frame=False,
        engine_kwargs={"return_profile": True},
    )

    result = run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([0, 1, 2, 3]),
        halfset_ids=np.asarray([0, 1, 0, 1], dtype=np.int8),
    )

    assert result.meta["halfset_0_profile_summary"] == {"em_time_s": 1.25, "batches": 1}
    assert result.meta["halfset_1_profile_summary"] == {"em_time_s": 1.25, "batches": 1}
    assert "fused_profile_summary" not in result.meta


def test_dense_initial_model_estep_uses_current_state_reference_when_means_omitted(monkeypatch):
    calls = []

    def fake_reference_to_dense_means(references):
        refs = np.asarray(references)
        return np.full((refs.shape[0], refs.shape[1] ** 3), refs[0, 0, 0, 0], dtype=np.complex64)

    def fake_run_dense_k_class_em(
        dataset, means, mean_variance, noise_variance, rotations, translations, disc_type, **kwargs
    ):
        calls.append(
            {
                "means": np.asarray(means).copy(),
                "mean_variance": np.asarray(mean_variance).copy(),
            }
        )
        return _fake_result(n_classes=1, n=8)

    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.reference_to_dense_means",
        fake_reference_to_dense_means,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.run_dense_k_class_em",
        fake_run_dense_k_class_em,
    )
    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=1,
        nr_iter=1,
        n_directions=4,
        pseudo_halfsets=False,
    )
    state.Iref[0, 0, 0, 0] = 7.0
    config = DenseInitialModelEstepConfig(
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_bpref_frame=False,
    )

    run_dense_initial_model_estep(_Dataset(), state, config)

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0]["means"], 7.0)
    np.testing.assert_allclose(calls[0]["mean_variance"], 49.0)


def test_reference_to_dense_means_uses_scoring_fourier_scale(monkeypatch):
    def fake_dft3(values):
        return np.asarray(values) + 2.0j

    def fake_gridding_correct(values, *args, **kwargs):
        return np.asarray(values) + 1.0, None

    monkeypatch.setattr("recovar.core.fourier_transform_utils.get_dft3", fake_dft3)
    monkeypatch.setattr("recovar.reconstruction.relion_functions.griddingCorrect", fake_gridding_correct)

    refs = np.zeros((1, 4, 4, 4), dtype=np.float32)

    means = reference_to_dense_means(refs)

    assert means.shape == (1, 4**3)
    np.testing.assert_allclose(means, 1.0 + 2.0j)


def test_relion_projector_to_dense_volume_embeds_cropped_slab(monkeypatch):
    captured = {}

    def fake_half_to_full(half, shape):
        captured["half"] = np.asarray(half)
        captured["shape"] = shape
        return np.asarray(half) + 1.0j

    monkeypatch.setattr("recovar.core.fourier_transform_utils.half_volume_to_full_volume", fake_half_to_full)

    slab = np.arange(3 * 3 * 2, dtype=np.float64).reshape(3, 3, 2).astype(np.complex128)
    out = _relion_projector_to_dense_volume(slab, 4)

    assert captured["shape"] == (4, 4, 4)
    half = captured["half"]
    assert half.shape == (4, 4, 3)
    np.testing.assert_array_equal(half[1:4, 1:4, :2], slab[::-1, :, :])
    np.testing.assert_allclose(out, half + 1.0j)


def test_relion_projector_to_dense_volume_handles_ori_size_boundary(monkeypatch):
    """When current_size == ori_size, RELION's cropped projector has y/z dim
    2*r_max+1 = ori_size+1. The embedding loop must drop the redundant
    Nyquist row (Hermitian conjugate of index 0) without raising."""
    captured = {}

    def fake_half_to_full(half, shape):
        captured["half"] = np.asarray(half)
        return np.asarray(half)

    monkeypatch.setattr("recovar.core.fourier_transform_utils.half_volume_to_full_volume", fake_half_to_full)

    # ori_size=4 → r_max=2 → cropped shape (5, 5, 3)
    slab = np.arange(5 * 5 * 3, dtype=np.float64).reshape(5, 5, 3).astype(np.complex128)
    out = _relion_projector_to_dense_volume(slab, 4)

    half = captured["half"]
    assert half.shape == (4, 4, 3)
    # Index iz=4 (extra Nyquist) must be dropped, not raise.
    # The first 4 rows (iz=0..3) of the reversed slab map to half[0..3, :, :].
    np.testing.assert_array_equal(half[0:4, 0:4, :3], slab[::-1, :, :][0:4, 0:4, :3])


def test_relion_projector_to_dense_volume_rejects_oversize(monkeypatch):
    """Beyond ori_size+1 in y/z (or center+1 in x) is a real shape error."""
    monkeypatch.setattr(
        "recovar.core.fourier_transform_utils.half_volume_to_full_volume",
        lambda half, shape: np.asarray(half),
    )
    slab = np.zeros((6, 6, 3), dtype=np.complex128)  # ori_size=4 → max 5x5x3
    with pytest.raises(ValueError, match="does not fit ori_size=4"):
        _relion_projector_to_dense_volume(slab, 4)


def test_reference_to_relion_projector_dense_means_uses_relion_projector_frame(monkeypatch):
    calls = []

    def fake_recovar_volume_to_relion(ref):
        return np.asarray(ref) + 10.0

    def fake_compute_fourier_transform_map(
        vol, ori_size, padding_factor, interpolator, current_size, do_gridding, data_dim
    ):
        calls.append(
            {
                "vol": np.asarray(vol).copy(),
                "ori_size": ori_size,
                "padding_factor": padding_factor,
                "interpolator": interpolator,
                "current_size": current_size,
                "do_gridding": do_gridding,
                "data_dim": data_dim,
            }
        )
        return np.ones((3, 3, 2), dtype=np.complex128), np.zeros(1), ori_size, padding_factor, 1, 0, interpolator

    def fake_embed(projector_data, ori_size):
        assert projector_data.shape == (3, 3, 2)
        return np.full((ori_size, ori_size, ori_size), 2.0 + 3.0j, dtype=np.complex128)

    monkeypatch.setattr("recovar.utils.helpers.recovar_volume_to_relion", fake_recovar_volume_to_relion)
    monkeypatch.setattr(
        "recovar.relion_bind._relion_bind_core.compute_fourier_transform_map",
        fake_compute_fourier_transform_map,
    )
    monkeypatch.setattr("recovar.em.initial_model.dense_adapter._relion_projector_to_dense_volume", fake_embed)

    refs = np.zeros((1, 4, 4, 4), dtype=np.float32)
    means = reference_to_relion_projector_dense_means(refs, current_size=2, padding_factor=1)

    assert means.shape == (1, 4**3)
    assert means.dtype == np.complex64
    np.testing.assert_allclose(means, -16.0 * (2.0 + 3.0j))
    assert len(calls) == 1
    call = calls[0]
    np.testing.assert_allclose(call["vol"], np.full((4, 4, 4), 10.0, dtype=np.float64))
    assert {k: v for k, v in call.items() if k != "vol"} == {
        "ori_size": 4,
        "padding_factor": 1,
        "interpolator": 1,
        "current_size": 2,
        "do_gridding": True,
        "data_dim": 2,
    }


def test_dense_initial_model_estep_handles_empty_halfset(monkeypatch):
    calls = []

    def fake_run_dense_k_class_em(*args, **kwargs):
        calls.append(kwargs["image_indices"])
        result = _fake_result(n_classes=1, n=8, n_images=int(np.asarray(kwargs["image_indices"]).size), n_groups=2)
        result.grouped_Ft_y[1] = 0.0
        result.grouped_Ft_ctf[1] = 0.0
        return result

    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.run_dense_k_class_em",
        fake_run_dense_k_class_em,
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4)
    config = DenseInitialModelEstepConfig(
        means=np.zeros((1, 8**3), dtype=np.complex64),
        mean_variance=np.ones((1, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_bpref_frame=False,
    )

    result = run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([0, 2]),
        halfset_ids=np.asarray([0, 0], dtype=np.int8),
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [0, 2])
    assert len(result.accumulators) == 2
    np.testing.assert_allclose(result.accumulators[1].data, 0.0)
    np.testing.assert_allclose(result.accumulators[1].weight, 0.0)


def test_dense_initial_model_estep_sparse_pass2_uses_coarse_parent_prior(monkeypatch):
    calls = {}

    def fake_significance(dataset, means, noise_variance, rotations, translations, disc_type, **kwargs):
        del means, noise_variance, disc_type
        calls["pass1_rotations"] = np.asarray(rotations, dtype=np.float32).copy()
        calls["pass1_translations"] = np.asarray(translations).copy()
        calls["pass1_prior"] = np.asarray(kwargs["translation_log_prior"], dtype=np.float32).copy()
        calls["pass1_current_size"] = kwargs["current_size"]
        n_images = int(dataset.n_images)
        n_rot = int(np.asarray(rotations).shape[0])
        significant = [[np.array([0], dtype=np.int32) for _ in range(n_images)]]
        return (
            np.ones((1, n_rot), dtype=bool),
            np.ones(n_images, dtype=np.int32),
            np.zeros(n_images, dtype=np.int32),
            np.zeros(n_images, dtype=np.int32),
            significant,
            None,
        )

    def fake_build_layout(*args, **kwargs):
        calls["pass2_parent_prior"] = np.asarray(kwargs["translation_log_prior"], dtype=np.float32).copy()
        calls["fine_prior"] = kwargs["fine_translation_log_prior"]
        calls["layout_translations"] = np.asarray(args[4], dtype=np.float32).copy() if args else None
        return LocalHypothesisLayout(
            n_global_rotations=1,
            n_pixels=1,
            n_psi=1,
            rotation_offsets=np.array([0, 1, 2], dtype=np.int64),
            rotation_ids_flat=np.array([0, 0], dtype=np.int32),
            rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy(),
            rotation_log_priors_flat=np.zeros(2, dtype=np.float32),
            rotation_counts=np.array([1, 1], dtype=np.int32),
            translation_grid=np.zeros((4, 2), dtype=np.float32),
            translation_log_priors=np.zeros((2, 4), dtype=np.float32),
            rotation_posterior_ids_flat=np.array([0, 0], dtype=np.int32),
        )

    def fake_run_local(dataset, means, mean_variance, noise_variance, local_layout, disc_type, **kwargs):
        del means, mean_variance, noise_variance, disc_type
        calls["local_n_global_rotations"] = int(local_layout.n_global_rotations)
        calls["local_pre_shifts"] = np.asarray(kwargs["image_pre_shifts"], dtype=np.float32).copy()
        calls["local_current_size"] = kwargs["current_size"]
        calls["local_has_reconstruct_with_masked_images"] = "reconstruct_with_masked_images" in kwargs
        calls["local_has_reconstruction_subtract_projected_reference"] = (
            "reconstruction_subtract_projected_reference" in kwargs
        )
        calls["local_has_recon_square_window"] = "recon_square_window" in kwargs
        calls["local_has_recon_exact_radius"] = "recon_exact_radius" in kwargs
        calls["local_mstep_subtract_ctf_projection"] = kwargs["mstep_subtract_ctf_projection"]
        calls["local_mstep_relion_x_half"] = kwargs["mstep_relion_x_half"]
        return _fake_result(n_classes=1, n=8, n_images=int(dataset.n_units), n_groups=1)

    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter._compute_k_class_significance_batched",
        fake_significance,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.build_pass2_hypothesis_layout",
        fake_build_layout,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.run_local_k_class_em",
        fake_run_local,
    )

    def fake_perturb(rotations, random_perturbation, angular_sampling_deg):
        calls["rotation_perturbation"] = (float(random_perturbation), float(angular_sampling_deg))
        return np.asarray(rotations, dtype=np.float32) + np.float32(7.0)

    monkeypatch.setattr(
        "recovar.em.sampling.apply_relion_rotation_perturbation",
        fake_perturb,
    )

    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=1,
        nr_iter=1,
        n_directions=4,
        pseudo_halfsets=False,
    )
    fine_prior = np.arange(16, dtype=np.float32).reshape(4, 4)
    coarse_prior = np.arange(8, dtype=np.float32).reshape(4, 2)
    pre_shifts = np.arange(8, dtype=np.float32).reshape(4, 2)
    coarse_translations = np.asarray([[0.25, 0.25], [1.25, 0.25]], dtype=np.float32)
    config = DenseInitialModelEstepConfig(
        means=np.zeros((1, 8**3), dtype=np.complex64),
        mean_variance=np.ones((1, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.zeros((12, 3, 3), dtype=np.float32),
        translations=np.zeros((4, 2), dtype=np.float32),
        relion_bpref_frame=False,
        engine_kwargs={
            "sparse_pass2": True,
            "healpix_order": 0,
            "oversampling_order": 1,
            "translation_step": 1.0,
            "random_perturbation": 0.25,
            "coarse_translations": coarse_translations,
            "pass1_current_size": 6,
            "translation_log_prior": fine_prior,
            "coarse_translation_log_prior": coarse_prior,
            "image_pre_shifts": pre_shifts,
            "reconstruct_with_masked_images": True,
            "reconstruction_subtract_projected_reference": True,
        },
    )

    result = run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([1, 3], dtype=np.int64),
    )

    np.testing.assert_allclose(calls["pass1_translations"], coarse_translations)
    np.testing.assert_allclose(calls["pass1_prior"], coarse_prior[[1, 3]])
    assert calls["pass1_current_size"] == 6
    assert calls["local_current_size"] == state.current_size
    assert calls["rotation_perturbation"] == (0.25, 60.0)
    assert np.max(calls["pass1_rotations"]) > 6.0
    np.testing.assert_allclose(calls["pass2_parent_prior"], coarse_prior[[1, 3]])
    assert calls["fine_prior"] is None
    np.testing.assert_allclose(calls["local_pre_shifts"], pre_shifts[[1, 3]])
    assert calls["local_n_global_rotations"] == 1
    assert calls["local_has_reconstruct_with_masked_images"] is False
    assert calls["local_has_reconstruction_subtract_projected_reference"] is False
    assert calls["local_has_recon_square_window"] is False
    assert calls["local_has_recon_exact_radius"] is False
    assert calls["local_mstep_subtract_ctf_projection"] is True
    assert calls["local_mstep_relion_x_half"] is False
    assert result.meta["sparse_pass2"] is True
    np.testing.assert_array_equal(result.meta["selected_particle_ids"], [1, 3])
    np.testing.assert_array_equal(result.meta["best_pose_rotation_ids"], [0, 1])
    np.testing.assert_allclose(result.meta["best_pose_translations"], [[0, 1], [2, 3]])


def test_dense_initial_model_estep_sparse_pass2_preserves_k_class_state(monkeypatch):
    calls = {}

    def fake_significance(dataset, means, noise_variance, rotations, translations, disc_type, **kwargs):
        del noise_variance, translations, disc_type
        calls["pass1_means_shape"] = np.asarray(means).shape
        calls["pass1_class_log_priors"] = np.asarray(kwargs["class_log_priors"], dtype=np.float64).copy()
        n_images = int(dataset.n_images)
        n_rot = int(np.asarray(rotations).shape[0])
        significant = [[np.asarray([class_idx], dtype=np.int32) for _ in range(n_images)] for class_idx in range(2)]
        return (
            np.ones((2, n_rot), dtype=bool),
            np.full(n_images, 2, dtype=np.int32),
            np.zeros(n_images, dtype=np.int32),
            np.zeros(n_images, dtype=np.int32),
            significant,
            None,
        )

    def fake_build_layout(significant_samples, *args, **kwargs):
        del args
        calls["significant_samples"] = [np.asarray(samples, dtype=np.int32).copy() for samples in significant_samples]
        calls["pass2_parent_prior"] = np.asarray(kwargs["translation_log_prior"], dtype=np.float32).copy()
        calls["fine_prior"] = kwargs["fine_translation_log_prior"]
        calls["rotation_index_order"] = kwargs["rotation_index_order"]
        return LocalHypothesisLayout(
            n_global_rotations=1,
            n_pixels=1,
            n_psi=1,
            rotation_offsets=np.array([0, 2, 4], dtype=np.int64),
            rotation_ids_flat=np.array([0, 0, 0, 0], dtype=np.int32),
            rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy(),
            rotation_log_priors_flat=np.zeros(4, dtype=np.float32),
            rotation_counts=np.array([2, 2], dtype=np.int32),
            translation_grid=np.zeros((2, 2), dtype=np.float32),
            translation_log_priors=np.zeros((2, 2), dtype=np.float32),
            rotation_posterior_ids_flat=np.array([0, 0, 0, 0], dtype=np.int32),
        )

    def fake_run_local(dataset, means, mean_variance, noise_variance, local_layout, disc_type, **kwargs):
        del noise_variance, local_layout, disc_type
        calls["pass2_means_shape"] = np.asarray(means).shape
        calls["pass2_mean_variance_shape"] = np.asarray(mean_variance).shape
        calls["pass2_class_log_priors"] = np.asarray(kwargs["class_log_priors"], dtype=np.float64).copy()
        return _fake_result(n_classes=2, n=8, n_images=int(dataset.n_units), n_groups=1)

    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter._compute_k_class_significance_batched",
        fake_significance,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.build_pass2_hypothesis_layout",
        fake_build_layout,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.run_local_k_class_em",
        fake_run_local,
    )

    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=2,
        nr_iter=1,
        n_directions=4,
        pseudo_halfsets=False,
    )
    state.pdf_class = np.asarray([0.8, 0.2])
    fine_prior = np.arange(16, dtype=np.float32).reshape(4, 4)
    config = DenseInitialModelEstepConfig(
        means=np.zeros((2, 8**3), dtype=np.complex64),
        mean_variance=np.ones((2, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.zeros((12, 3, 3), dtype=np.float32),
        translations=np.zeros((4, 2), dtype=np.float32),
        relion_bpref_frame=False,
        engine_kwargs={
            "sparse_pass2": True,
            "healpix_order": 0,
            "oversampling_order": 1,
            "translation_step": 1.0,
            "translation_log_prior": fine_prior,
        },
    )

    result = run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([1, 3], dtype=np.int64),
    )

    assert calls["pass1_means_shape"] == (2, 8**3)
    assert calls["pass2_means_shape"] == (2, 8**3)
    assert calls["pass2_mean_variance_shape"] == (2, 8**3)
    np.testing.assert_allclose(calls["pass1_class_log_priors"], np.log([0.8, 0.2]))
    np.testing.assert_allclose(calls["pass2_class_log_priors"], np.log([0.8, 0.2]))
    np.testing.assert_allclose(calls["pass2_parent_prior"], fine_prior[[1, 3]])
    assert calls["fine_prior"] is None
    assert calls["rotation_index_order"] == "relion_hidden"
    np.testing.assert_array_equal(calls["significant_samples"][0], [0, 1])
    np.testing.assert_array_equal(calls["significant_samples"][1], [0, 1])
    assert [(a.halfset_idx, a.class_idx) for a in result.accumulators] == [(0, 0), (0, 1)]
    np.testing.assert_allclose(result.meta["class_posterior_sums"], [0.0, 1.0])
    np.testing.assert_allclose(
        result.meta["class_direction_posterior_sums"],
        np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
    )
    np.testing.assert_array_equal(result.meta["selected_particle_ids"], [1, 3])
    np.testing.assert_array_equal(result.meta["class_assignments"], [0, 0])
    np.testing.assert_array_equal(result.meta["best_pose_rotation_ids"], [0, 1])
    assert result.meta["best_pose_rotations"].shape == (2, 3, 3)
    assert result.meta["sparse_pass2"] is True


def test_dense_initial_model_estep_sparse_pass2_pseudo_halfsets_use_separate_local_runs(monkeypatch):
    calls = {"significance": [], "layouts": [], "local": []}

    def fake_significance(dataset, means, noise_variance, rotations, translations, disc_type, **kwargs):
        del noise_variance, translations, disc_type
        calls["significance"].append(
            {
                "has_image_indices": "image_indices" in kwargs,
                "current_size": kwargs["current_size"],
                "n_images": int(dataset.n_images),
            }
        )
        n_images = int(dataset.n_images)
        n_rot = int(np.asarray(rotations).shape[0])
        significant = [[np.asarray([class_idx], dtype=np.int32) for _ in range(n_images)] for class_idx in range(1)]
        return (
            np.ones((1, n_rot), dtype=bool),
            np.full(n_images, 1, dtype=np.int32),
            np.zeros(n_images, dtype=np.int32),
            np.zeros(n_images, dtype=np.int32),
            significant,
            None,
        )

    def fake_build_layout(significant_samples, *args, **kwargs):
        del args
        calls["layouts"].append(
            {
                "significant_samples": [np.asarray(samples, dtype=np.int32).copy() for samples in significant_samples],
                "pass2_parent_prior": np.asarray(kwargs["translation_log_prior"], dtype=np.float32).copy(),
            }
        )
        return LocalHypothesisLayout(
            n_global_rotations=1,
            n_pixels=1,
            n_psi=1,
            rotation_offsets=np.array([0, 2, 4, 6, 8], dtype=np.int64),
            rotation_ids_flat=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
            rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (8, 3, 3)).copy(),
            rotation_log_priors_flat=np.zeros(8, dtype=np.float32),
            rotation_counts=np.array([2, 2, 2, 2], dtype=np.int32),
            translation_grid=np.zeros((2, 2), dtype=np.float32),
            translation_log_priors=np.zeros((4, 2), dtype=np.float32),
            rotation_posterior_ids_flat=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
        )

    def fake_run_local(dataset, means, mean_variance, noise_variance, local_layout, disc_type, **kwargs):
        del noise_variance, local_layout, disc_type
        calls["local"].append(
            {
                "has_image_indices": "image_indices" in kwargs,
                "has_reconstruction_group_ids": "reconstruction_group_ids" in kwargs,
                "has_reconstruction_group_count": "reconstruction_group_count" in kwargs,
                "n_images": int(dataset.n_images),
            }
        )
        return _fake_result(n_classes=1, n=8, n_images=int(dataset.n_images), n_groups=1)

    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter._compute_k_class_significance_batched",
        fake_significance,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.build_pass2_hypothesis_layout",
        fake_build_layout,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.run_local_k_class_em",
        fake_run_local,
    )

    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=1,
        nr_iter=1,
        n_directions=4,
        pseudo_halfsets=True,
    )
    fine_prior = np.arange(16, dtype=np.float32).reshape(4, 4)
    config = DenseInitialModelEstepConfig(
        means=np.zeros((1, 8**3), dtype=np.complex64),
        mean_variance=np.ones((1, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.zeros((12, 3, 3), dtype=np.float32),
        translations=np.zeros((4, 2), dtype=np.float32),
        relion_bpref_frame=False,
        engine_kwargs={
            "sparse_pass2": True,
            "healpix_order": 0,
            "oversampling_order": 1,
            "translation_step": 1.0,
            "translation_log_prior": fine_prior,
        },
    )

    result = run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([0, 1, 2, 3], dtype=np.int64),
        halfset_ids=np.asarray([0, 1, 0, 1], dtype=np.int8),
    )

    assert calls["significance"] == [
        {"has_image_indices": False, "current_size": state.current_size, "n_images": 2},
        {"has_image_indices": False, "current_size": state.current_size, "n_images": 2},
    ]
    assert len(calls["layouts"]) == 2
    np.testing.assert_array_equal(calls["layouts"][0]["pass2_parent_prior"], fine_prior[[0, 2]])
    np.testing.assert_array_equal(calls["layouts"][1]["pass2_parent_prior"], fine_prior[[1, 3]])
    np.testing.assert_array_equal(calls["layouts"][0]["significant_samples"][0], [0])
    np.testing.assert_array_equal(calls["layouts"][0]["significant_samples"][1], [0])
    np.testing.assert_array_equal(calls["layouts"][1]["significant_samples"][0], [0])
    np.testing.assert_array_equal(calls["layouts"][1]["significant_samples"][1], [0])
    assert calls["local"] == [
        {
            "has_image_indices": False,
            "has_reconstruction_group_ids": False,
            "has_reconstruction_group_count": False,
            "n_images": 2,
        },
        {
            "has_image_indices": False,
            "has_reconstruction_group_ids": False,
            "has_reconstruction_group_count": False,
            "n_images": 2,
        },
    ]
    np.testing.assert_array_equal(result.meta["selected_particle_ids"], [0, 2, 1, 3])
    np.testing.assert_array_equal(result.meta["best_pose_rotation_ids"], [0, 1, 0, 1])
    assert "fused_pseudo_halfsets" not in result.meta


def test_sparse_pass2_pass1_current_size_matches_relion_fixture_coarse_size():
    state = initialise_denovo_state(
        ori_size=64,
        pixel_size=8.5,
        K=1,
        nr_iter=1,
        n_directions=576,
        pseudo_halfsets=False,
    )
    assert state.current_size == 28

    pass1_current_size = _resolve_sparse_pass1_current_size(
        state,
        {"current_size": state.current_size},
        {"healpix_order": 1, "particle_diameter_ang": 544.0},
    )

    assert pass1_current_size == 10


def test_initial_model_pass2_layout_uses_relion_direction_ids_for_posterior_bins():
    layout = LocalHypothesisLayout(
        n_global_rotations=4,
        n_pixels=12,
        n_psi=2,
        rotation_offsets=np.array([0, 4], dtype=np.int64),
        rotation_ids_flat=np.array([4, 8, 9, 10], dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(4, dtype=np.float32),
        rotation_counts=np.array([4], dtype=np.int32),
        translation_grid=np.zeros((2, 2), dtype=np.float32),
        translation_log_priors=np.zeros((1, 2), dtype=np.float32),
        rotation_posterior_ids_flat=np.array([0, 1, 2, 3], dtype=np.int32),
        sample_mask_flat=np.ones((4, 2), dtype=bool),
    )

    out = _initial_model_pass2_layout(layout)

    assert out.n_global_rotations == 2
    np.testing.assert_array_equal(out.rotation_posterior_ids_flat, np.array([0, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(out.rotation_ids_flat, layout.rotation_ids_flat)
