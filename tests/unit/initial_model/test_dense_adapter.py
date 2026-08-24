"""InitialModel dense K-class E-step adapter tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume.local_layout import LocalHypothesisLayout
from recovar.em.initial_model import initialise_denovo_state
from recovar.em.initial_model.dense_adapter import (
    DenseInitialModelEstepConfig,
    _arrays_to_accumulators,
    _estep_meta,
    _initial_model_pass2_layout,
    _relion_projector_to_dense_volume,
    _resolve_class_inputs,
    _resolve_sparse_pass1_current_size,
    _safe_coarse_significance_image_batch_size,
    class_log_priors_from_state,
    reference_to_dense_means,
    reference_to_relion_projector_dense_means,
    run_dense_initial_model_estep,
    split_pseudo_halfset_particle_ids,
)

pytestmark = pytest.mark.unit


def test_coarse_significance_batch_is_unchanged_for_small_pose_grids():
    assert _safe_coarse_significance_image_batch_size(
        500,
        n_classes=1,
        n_rotations=4608,
        n_translations=45,
    ) == 500


@pytest.mark.parametrize(
    ("n_translations", "expected"),
    [(37, 146), (45, 120)],
)
def test_coarse_significance_batch_caps_gui_default_oom_grids(
    n_translations,
    expected,
):
    assert _safe_coarse_significance_image_batch_size(
        500,
        n_classes=1,
        n_rotations=36864,
        n_translations=n_translations,
    ) == expected


def test_coarse_significance_batch_respects_smaller_user_request_and_k_axis():
    assert _safe_coarse_significance_image_batch_size(
        64,
        n_classes=1,
        n_rotations=36864,
        n_translations=45,
    ) == 64
    assert _safe_coarse_significance_image_batch_size(
        500,
        n_classes=4,
        n_rotations=36864,
        n_translations=45,
    ) == 30


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


def test_arrays_to_accumulators_inverts_relion_x_public_layout_without_projector_flip():
    from recovar.em.dense_single_volume.helpers.half_volume_mstep import (
        relion_x_half_volume_to_full,
    )
    from recovar.em.dense_single_volume.local_backprojection import (
        enforce_relion_half_volume_x0_hermitian_host,
    )
    from recovar.em.initial_model.layout import relion_bpref_frame_scales

    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=1,
        nr_iter=1,
        n_directions=4,
    )
    state.current_size = 4
    compact_shape = (7, 7, 7)
    half_shape = (7, 7, 4)
    rng = np.random.default_rng(93)
    bp_data = (
        rng.standard_normal(half_shape) + 1j * rng.standard_normal(half_shape)
    ).astype(np.complex64)
    bp_weight = rng.uniform(1e-3, 2.0, size=half_shape).astype(np.float32)
    bp_data = enforce_relion_half_volume_x0_hermitian_host(
        bp_data.reshape(-1), compact_shape
    ).reshape(half_shape)
    bp_weight = enforce_relion_half_volume_x0_hermitian_host(
        bp_weight.reshape(-1), compact_shape
    ).reshape(half_shape)
    public_data = relion_x_half_volume_to_full(bp_data.reshape(-1), compact_shape)
    public_weight = relion_x_half_volume_to_full(bp_weight.reshape(-1), compact_shape)

    actual = _arrays_to_accumulators(
        [public_data],
        [public_weight],
        state,
        halfset_idx=0,
        relion_bpref_frame=True,
        relion_projector_frame=True,
        padding_factor=1,
    )[0]

    data_scale, weight_scale = relion_bpref_frame_scales(state.ori_size)
    np.testing.assert_array_equal(actual.data, bp_data.astype(np.complex128) * data_scale)
    np.testing.assert_array_equal(actual.weight, bp_weight.astype(np.float64) * weight_scale)


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
            noise_stats=(
                _fake_noise_stats(0.0, 0.25, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                _fake_noise_stats(0.0, 0.75, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ),
            aggregate_noise_stats=_fake_noise_stats(10.0, 3.0, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        ),
        1: SimpleNamespace(
            class_posterior_sums=np.asarray([3.0, 4.0], dtype=np.float32),
            noise_stats=(
                _fake_noise_stats(0.0, 1.25, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                _fake_noise_stats(0.0, 1.75, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ),
            aggregate_noise_stats=_fake_noise_stats(20.0, 7.0, [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]),
        ),
    }

    meta = _estep_meta(halfset_results)

    assert meta["wsum_sigma2_offset"] == pytest.approx(30.0)
    assert meta["sigma2_offset_sumw"] == pytest.approx(10.0)
    assert meta["noise_sumw"] == pytest.approx(10.0)
    np.testing.assert_allclose(meta["class_reconstruction_support_sums"], [1.5, 2.5])
    np.testing.assert_allclose(meta["halfset_0_class_reconstruction_support_sums"], [0.25, 0.75])
    np.testing.assert_allclose(meta["halfset_1_class_reconstruction_support_sums"], [1.25, 1.75])
    np.testing.assert_allclose(meta["wsum_sigma2_noise"], [8.0, 10.0, 12.0])
    np.testing.assert_allclose(meta["wsum_img_power"], [14.0, 16.0, 18.0])
    np.testing.assert_allclose(meta["halfset_0_wsum_sigma2_noise"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(meta["halfset_1_wsum_img_power"], [10.0, 11.0, 12.0])


def test_estep_meta_uses_significant_mstep_mass_for_relion_probability_updates():
    halfset_results = {
        0: SimpleNamespace(
            class_posterior_sums=np.asarray([1.0, 2.0], dtype=np.float32),
            class_mstep_posterior_sums=np.asarray([0.8, 1.9], dtype=np.float32),
        ),
        1: SimpleNamespace(
            class_posterior_sums=np.asarray([3.0, 4.0], dtype=np.float32),
            class_mstep_posterior_sums=np.asarray([2.7, 3.6], dtype=np.float32),
        ),
    }

    meta = _estep_meta(halfset_results)

    np.testing.assert_allclose(meta["class_posterior_sums"], [3.5, 5.5])
    np.testing.assert_allclose(meta["class_posterior_sums_full"], [4.0, 6.0])
    np.testing.assert_allclose(meta["halfset_0_class_posterior_sums"], [0.8, 1.9])
    np.testing.assert_allclose(meta["halfset_0_class_posterior_sums_full"], [1.0, 2.0])


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
        # ``return_profile`` is filtered out by _dense_run_em_kwargs before
        # reaching run_dense_k_class_em — it's a sparse/local-engine-only
        # kwarg that run_dense_k_class_em explicitly _reject_kwargs's.
        # dense_adapter still extracts profile info from the return-value
        # meta regardless of whether the kwarg flowed through.
        assert "return_profile" not in kwargs
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
        # See note in test_dense_initial_model_estep_meta_includes_optional_profiles
        # — return_profile is filtered out before the dense entry-point.
        assert "return_profile" not in kwargs
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


def test_relion_projector_to_dense_volume_truncates_oversize(monkeypatch):
    """Slabs larger than the representable half-volume are truncated to the
    in-range subset rather than rejected. This handles VDAM iters where
    autosampling pushes current_size up to (or slightly above) ori_size and
    RELION emits cropped projectors of shape (2*r_max+1, 2*r_max+1, r_max+1)
    that exceed RECOVAR's (ori_size, ori_size, ori_size/2+1) layout."""
    captured = {}

    def fake_half_to_full(half, shape):
        captured["half"] = np.asarray(half)
        return np.asarray(half)

    monkeypatch.setattr("recovar.core.fourier_transform_utils.half_volume_to_full_volume", fake_half_to_full)
    # ori_size=4 → max half (4, 4, 3). Pass an even larger (7, 7, 4) slab.
    slab = np.arange(7 * 7 * 4, dtype=np.float64).reshape(7, 7, 4).astype(np.complex128)
    out = _relion_projector_to_dense_volume(slab, 4)
    half = captured["half"]
    assert half.shape == (4, 4, 3)
    # Center of slab (index 3) maps to center of half (index 2).
    # iz=3 → z = 3-3+2 = 2 ✓, iz=2 → z = 1, iz=4 → z = 3, iz=0/1/5/6 → out of range.
    # Reversed slab[::-1] at iz=2 = original slab[4]; at iz=3 = slab[3]; etc.
    rev = slab[::-1, :, :]
    np.testing.assert_array_equal(half[1, 1, :3], rev[2, 2, :3])
    np.testing.assert_array_equal(half[2, 2, :3], rev[3, 3, :3])


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


def test_relion_projector_projection_dense_scale_matches_embedded_means(monkeypatch):
    import jax.numpy as jnp

    from recovar.em.dense_single_volume.helpers import projection as projection_helpers

    raw = jnp.asarray([[1.0 + 2.0j, -3.0 + 0.5j]], dtype=jnp.complex64)

    def fake_project(*args, **kwargs):
        return raw

    monkeypatch.setattr(
        projection_helpers,
        "project_relion_projector_half_spectrum_centered_rows",
        fake_project,
    )

    rotations = np.eye(3, dtype=np.float32)[None]
    proj, proj_abs2 = projection_helpers.compute_relion_projector_projections_block(
        np.zeros((3, 3, 2), dtype=np.complex64),
        rotations,
        (4, 4),
        r_max=1,
        centered_rows=True,
        dense_scale=True,
    )

    expected = np.asarray(raw) * -16.0
    # float32 round-trip through jnp / jax has ~1 ULP relative error; the
    # default rtol=1e-7 of assert_allclose is too tight for float32.
    np.testing.assert_allclose(np.asarray(proj), expected, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(np.asarray(proj_abs2), np.abs(expected) ** 2, rtol=1e-5, atol=1e-3)


def test_resolve_class_inputs_relion_projector_uses_exact_path_by_default(monkeypatch):
    projector_half = np.ones((1, 3, 3, 2), dtype=np.complex64)
    dense_means = np.full((1, 8**3), 2.0 + 0.5j, dtype=np.complex64)

    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.reference_to_relion_projector_half_maps",
        lambda *args, **kwargs: (projector_half, 2),
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.relion_projector_half_maps_to_dense_means",
        lambda *args, **kwargs: dense_means,
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4)
    config = DenseInitialModelEstepConfig(
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_projector_frame=True,
    )

    means, _mean_variance, exact_half, exact_rmax = _resolve_class_inputs(state, config)

    np.testing.assert_array_equal(means, dense_means)
    np.testing.assert_array_equal(exact_half, projector_half)
    assert exact_rmax == 2

    monkeypatch.setenv("RECOVAR_INITIAL_MODEL_EXACT_RELION_PROJECTOR", "0")
    _means, _mean_variance, exact_half, exact_rmax = _resolve_class_inputs(state, config)

    assert exact_half is None
    assert exact_rmax is None


def test_resolve_class_inputs_can_dump_exact_projector_operand(monkeypatch, tmp_path):
    projector_half = np.arange(54, dtype=np.float32).reshape(1, 3, 3, 6)[..., :2].astype(np.complex64)
    dense_means = np.zeros((1, 8**3), dtype=np.complex64)
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.reference_to_relion_projector_half_maps",
        lambda *args, **kwargs: (projector_half, 2),
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.relion_projector_half_maps_to_dense_means",
        lambda *args, **kwargs: dense_means,
    )
    monkeypatch.setenv("RECOVAR_INITIAL_MODEL_PROJECTOR_DUMP_DIR", str(tmp_path))
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4)
    state.iter = 7
    state.current_size = 4
    config = DenseInitialModelEstepConfig(
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_projector_frame=True,
    )

    _resolve_class_inputs(state, config)

    with np.load(tmp_path / "iter007_relion_projector_half.npz") as dumped:
        np.testing.assert_array_equal(dumped["projector_half"], projector_half)
        assert int(dumped["projector_r_max"]) == 2
        assert int(dumped["current_size"]) == 4
        assert int(dumped["iteration"]) == 7


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
    monkeypatch.delenv("RECOVAR_INITIAL_MODEL_EXACT_FINE_DIFF2", raising=False)

    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as sparse_diagnostics

    monkeypatch.setattr(
        sparse_diagnostics,
        "set_bpref_contribution_dump_context",
        lambda **kwargs: calls.setdefault("diagnostic_context", []).append(kwargs),
    )
    monkeypatch.setattr(
        sparse_diagnostics,
        "clear_bpref_contribution_dump_context",
        lambda: calls.setdefault("diagnostic_context", []).append("clear"),
    )

    def fake_significance(dataset, means, noise_variance, rotations, translations, disc_type, **kwargs):
        del means, noise_variance, disc_type
        calls["pass1_rotations"] = np.asarray(rotations, dtype=np.float32).copy()
        calls["pass1_translations"] = np.asarray(translations).copy()
        calls["pass1_prior"] = np.asarray(kwargs["translation_log_prior"], dtype=np.float32).copy()
        calls["pass1_current_size"] = kwargs["current_size"]
        calls["pass1_max_significants"] = kwargs["max_significants"]
        calls["pass1_image_batch_size"] = kwargs["image_batch_size"]
        calls["pass1_debug_iteration"] = kwargs["debug_iteration"]
        calls["pass1_relion_coarse_gaussian_default"] = kwargs[
            "relion_coarse_gaussian_default"
        ]
        calls["pass1_pad_final_image_batch"] = kwargs["pad_final_image_batch"]
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
        assert isinstance(local_layout, tuple)
        calls["local_layout_count"] = len(local_layout)
        calls["local_n_global_rotations"] = int(local_layout[0].n_global_rotations)
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
        calls["local_max_significants"] = kwargs["max_significants"]
        calls["local_debug_iteration"] = kwargs["debug_iteration"]
        calls["local_use_float64_scoring"] = kwargs["use_float64_scoring"]
        calls["local_use_float64_normalization"] = kwargs["use_float64_normalization"]
        calls["local_unify_bucket_sizes"] = kwargs["unify_local_bucket_sizes"]
        calls["local_stats_use_reconstruction_probs"] = kwargs["stats_use_reconstruction_probs"]
        calls["local_class_posterior_sums_from_noise"] = kwargs["class_posterior_sums_from_noise"]
        calls["local_relion_f32_fine_posterior"] = kwargs["relion_f32_fine_posterior"]
        calls["local_projection_mask_current_image_disk"] = kwargs["projection_mask_current_image_disk"]
        calls["local_relion_exact_bpref_operands"] = kwargs[
            "relion_exact_bpref_operands"
        ]
        calls["local_preserve_bpref_particle_order"] = kwargs[
            "preserve_bpref_particle_order"
        ]
        calls["local_relion_exact_fine_diff2"] = kwargs["relion_exact_fine_diff2"]
        calls["local_relion_exact_score_translation"] = kwargs[
            "relion_exact_score_translation"
        ]
        return _fake_result(n_classes=1, n=8, n_images=int(dataset.n_units), n_groups=1)

    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter._compute_k_class_significance_batched",
        fake_significance,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter._safe_coarse_significance_image_batch_size",
        lambda *_args, **_kwargs: 7,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.build_pass2_hypothesis_layout",
        fake_build_layout,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter.run_local_k_class_em",
        fake_run_local,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter._resolve_class_inputs",
        lambda state, config: (
            config.means,
            config.mean_variance,
            np.zeros((1, 1), dtype=np.complex64),
            1,
        ),
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter._uses_relion_cuda_image_preprocessing",
        lambda dataset: True,
    )

    def fake_perturb(rotations, random_perturbation, angular_sampling_deg):
        calls["rotation_perturbation"] = (float(random_perturbation), float(angular_sampling_deg))
        return np.asarray(rotations, dtype=np.float32) + np.float32(7.0)

    monkeypatch.setattr(
        "recovar.em.sampling.apply_relion_rotation_perturbation",
        fake_perturb,
    )

    def fake_device_coarse(source_eulers, random_perturbation, angular_sampling_deg):
        source_eulers = np.asarray(source_eulers)
        calls["device_coarse"] = (
            source_eulers.copy(),
            float(random_perturbation),
            float(angular_sampling_deg),
        )
        return np.full((source_eulers.shape[0], 3, 3), 9.0, dtype=np.float32)

    monkeypatch.setattr(
        "recovar.em.sampling._relion_adaptive_pass1_rotations_f32",
        fake_device_coarse,
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
            "max_significants": 100,
            "debug_iteration": 7,
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
    assert calls["pass1_max_significants"] == 100
    assert calls["pass1_image_batch_size"] == 7
    assert calls["pass1_debug_iteration"] == 7
    assert calls["pass1_relion_coarse_gaussian_default"] is True
    assert calls["pass1_pad_final_image_batch"] is True
    assert calls["local_current_size"] == state.current_size
    assert calls["rotation_perturbation"] == (0.25, 60.0)
    assert calls["device_coarse"][0].shape == (72, 3)
    assert calls["device_coarse"][1:] == (0.25, 60.0)
    np.testing.assert_array_equal(calls["pass1_rotations"], np.full((72, 3, 3), 9.0, dtype=np.float32))
    np.testing.assert_allclose(calls["pass2_parent_prior"], coarse_prior[[1, 3]])
    assert calls["fine_prior"] is None
    np.testing.assert_allclose(calls["local_pre_shifts"], pre_shifts[[1, 3]])
    assert calls["local_layout_count"] == 1
    assert calls["local_n_global_rotations"] == 1
    assert calls["local_has_reconstruct_with_masked_images"] is False
    assert calls["local_has_reconstruction_subtract_projected_reference"] is False
    assert calls["local_has_recon_square_window"] is False
    assert calls["local_has_recon_exact_radius"] is False
    assert calls["local_mstep_subtract_ctf_projection"] is True
    assert calls["local_mstep_relion_x_half"] is False
    assert calls["local_max_significants"] == -1
    assert calls["local_debug_iteration"] == 7
    assert calls["local_use_float64_scoring"] is False
    assert calls["local_use_float64_normalization"] is True
    assert calls["local_unify_bucket_sizes"] is True
    assert calls["local_stats_use_reconstruction_probs"] is True
    assert calls["local_class_posterior_sums_from_noise"] is False
    assert calls["local_relion_f32_fine_posterior"] is False
    assert calls["local_projection_mask_current_image_disk"] is False
    assert calls["local_relion_exact_bpref_operands"] is True
    assert calls["local_preserve_bpref_particle_order"] is False
    assert calls["local_relion_exact_fine_diff2"] is True
    assert calls["local_relion_exact_score_translation"] is True
    assert calls["diagnostic_context"] == [{"iteration": 7, "half": 1}, "clear"]
    assert result.meta["sparse_pass2"] is True
    np.testing.assert_array_equal(result.meta["selected_particle_ids"], [1, 3])
    np.testing.assert_array_equal(result.meta["best_pose_rotation_ids"], [0, 1])
    np.testing.assert_allclose(result.meta["best_pose_translations"], [[0, 1], [2, 3]])


def test_exact_relion_fine_diff2_can_be_disabled(monkeypatch):
    from recovar.em.initial_model.dense_adapter import _exact_relion_fine_diff2_enabled

    for value in ("0", "false", "NO", "Off"):
        monkeypatch.setenv("RECOVAR_INITIAL_MODEL_EXACT_FINE_DIFF2", value)
        assert _exact_relion_fine_diff2_enabled() is False


def test_dense_initial_model_estep_os0_uses_device_coarse_rotations(monkeypatch):
    """Equal coarse/fine grid sizes must not bypass AccProjectorPlan arithmetic."""

    calls = {}
    host_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 72, axis=0)
    device_rotations = np.full((72, 3, 3), np.float32(9.0))

    class BoundaryReached(RuntimeError):
        pass

    def fake_device_coarse(source_eulers, random_perturbation, angular_sampling_deg):
        calls["source_eulers"] = np.asarray(source_eulers).copy()
        calls["random_perturbation"] = float(random_perturbation)
        calls["angular_sampling_deg"] = float(angular_sampling_deg)
        return device_rotations

    def fake_significance(_dataset, _means, _noise, rotations, *_args, **_kwargs):
        calls["pass1_rotations"] = np.asarray(rotations).copy()
        raise BoundaryReached

    monkeypatch.setattr(
        "recovar.em.sampling._relion_adaptive_pass1_rotations_f32",
        fake_device_coarse,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter._compute_k_class_significance_batched",
        fake_significance,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter._resolve_class_inputs",
        lambda state, config: (
            config.means,
            config.mean_variance,
            np.zeros((1, 1, 1, 1), dtype=np.complex64),
            1,
        ),
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter._uses_relion_cuda_image_preprocessing",
        lambda _dataset: True,
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
        rotations=host_rotations,
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_bpref_frame=False,
        engine_kwargs={
            "sparse_pass2": True,
            "healpix_order": 0,
            "oversampling_order": 0,
            "random_perturbation": 0.25,
            "coarse_translations": np.zeros((1, 2), dtype=np.float32),
        },
    )

    with pytest.raises(BoundaryReached):
        run_dense_initial_model_estep(
            _Dataset(),
            state,
            config,
            particle_ids=np.asarray([0, 1], dtype=np.int64),
        )

    assert calls["source_eulers"].shape == (72, 3)
    assert calls["random_perturbation"] == 0.25
    assert calls["angular_sampling_deg"] == 60.0
    np.testing.assert_array_equal(calls["pass1_rotations"], device_rotations)


def test_initial_model_local_bucket_unification_can_be_disabled(monkeypatch):
    from recovar.em.initial_model.dense_adapter import _unify_local_bucket_sizes_enabled

    monkeypatch.delenv("RECOVAR_INITIAL_MODEL_UNIFY_LOCAL_BUCKET_SIZES", raising=False)
    assert _unify_local_bucket_sizes_enabled() is True
    for value in ("0", "false", "NO", "Off"):
        monkeypatch.setenv("RECOVAR_INITIAL_MODEL_UNIFY_LOCAL_BUCKET_SIZES", value)
        assert _unify_local_bucket_sizes_enabled() is False


def test_dense_initial_model_estep_os0_keeps_coarse_normalization_pose_and_support(monkeypatch):
    from recovar.em.dense_single_volume.helpers.types import make_relion_stats
    from recovar.em.dense_single_volume.k_class import KClassEMResult

    calls = {}
    coarse_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 72, axis=0)
    coarse_rotations[:, 0, 0] = np.arange(72, dtype=np.float32)
    coarse_translations = np.asarray([[0.0, 0.0], [1.5, -0.5]], dtype=np.float32)
    coarse_hard = np.asarray([5, 6], dtype=np.int32)
    coarse_pmax = np.asarray([0.25, 0.75], dtype=np.float32)
    coarse_log_evidence = np.asarray([11.0, 12.0], dtype=np.float64)
    coarse_class_evidence = coarse_log_evidence[None, :]

    def fake_significance(dataset, *_args, **_kwargs):
        n_images = int(dataset.n_images)
        assert n_images == 2
        calls["pass1_current_size"] = _kwargs["current_size"]
        significant = [[np.asarray([5], dtype=np.int32), np.asarray([6], dtype=np.int32)]]
        return (
            np.ones((1, 72), dtype=bool),
            np.ones(n_images, dtype=np.int32),
            coarse_hard,
            np.zeros(n_images, dtype=np.int32),
            significant,
            {
                "normalization_log_z": np.asarray([3.0, 4.0], dtype=np.float64),
                "normalization_log_evidence": coarse_log_evidence,
                "log_evidence_per_image": coarse_log_evidence.astype(np.float32),
                "best_log_score_per_image": np.asarray([10.0, 11.5], dtype=np.float32),
                "max_posterior_per_image": coarse_pmax,
                "class_log_evidence_per_image": coarse_class_evidence,
            },
        )

    def fake_build_layout(*_args, **kwargs):
        assert kwargs["oversampling_order"] == 0
        return LocalHypothesisLayout(
            n_global_rotations=72,
            n_pixels=1,
            n_psi=6,
            rotation_offsets=np.asarray([0, 1, 2], dtype=np.int64),
            rotation_ids_flat=np.asarray([2, 3], dtype=np.int32),
            rotations_flat=coarse_rotations[[2, 3]],
            rotation_log_priors_flat=np.zeros(2, dtype=np.float32),
            rotation_counts=np.ones(2, dtype=np.int32),
            translation_grid=coarse_translations,
            translation_log_priors=np.zeros((2, 2), dtype=np.float32),
            rotation_posterior_ids_flat=np.asarray([2, 3], dtype=np.int32),
        )

    fine_stats = make_relion_stats(
        log_evidence_per_image=np.asarray([10.5, 11.5], dtype=np.float32),
        best_log_score_per_image=np.asarray([10.4, 11.4], dtype=np.float32),
        max_posterior_per_image=np.asarray([0.9, 0.8], dtype=np.float32),
        rotation_posterior_sums=np.ones(12, dtype=np.float32),
    )

    def fake_run_local(dataset, *_args, **kwargs):
        calls.update(kwargs)
        assert int(dataset.n_images) == 2
        return KClassEMResult(
            new_means=None,
            Ft_y=np.ones((1, 8**3), dtype=np.complex64),
            Ft_ctf=np.ones((1, 8**3), dtype=np.float32),
            per_class_hard_assignments=np.asarray([[0, 1]], dtype=np.int32),
            class_assignments=np.zeros(2, dtype=np.int32),
            pose_assignments=np.asarray([0, 1], dtype=np.int32),
            class_responsibilities=np.ones((1, 2), dtype=np.float32),
            class_posterior_sums=np.asarray([2.0], dtype=np.float32),
            stats=fine_stats,
            per_class_stats=(fine_stats,),
            noise_stats=None,
            aggregate_noise_stats=None,
            per_class_best_pose_rotations=(coarse_rotations[[0, 1]],),
            per_class_best_pose_translations=(coarse_translations[[0, 0]],),
            per_class_best_pose_rotation_ids=(np.asarray([0, 1], dtype=np.int32),),
            best_pose_rotations=coarse_rotations[[0, 1]],
            best_pose_translations=coarse_translations[[0, 0]],
            best_pose_rotation_ids=np.asarray([0, 1], dtype=np.int32),
        )

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
        pseudo_halfsets=False,
    )
    config = DenseInitialModelEstepConfig(
        means=np.zeros((1, 8**3), dtype=np.complex64),
        mean_variance=np.ones((1, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=coarse_rotations,
        translations=coarse_translations,
        relion_bpref_frame=False,
        engine_kwargs={
            "sparse_pass2": True,
            "healpix_order": 0,
            "oversampling_order": 0,
            "coarse_translations": coarse_translations,
        },
    )
    result = run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([0, 1], dtype=np.int64),
    )

    assert calls["pass1_current_size"] == state.current_size
    np.testing.assert_array_equal(calls["class_log_evidence"], coarse_class_evidence)
    np.testing.assert_array_equal(calls["normalization_max_posterior"], coarse_pmax)
    assert calls["reconstruct_significant_only"] is False
    np.testing.assert_array_equal(result.meta["pose_assignments"], coarse_hard)
    np.testing.assert_array_equal(result.meta["best_pose_rotation_ids"], [2, 3])
    np.testing.assert_array_equal(result.meta["best_pose_rotations"], coarse_rotations[[2, 3]])
    np.testing.assert_array_equal(result.meta["best_pose_translations"], coarse_translations[[1, 0]])
    np.testing.assert_array_equal(result.meta["max_posterior_per_image"], coarse_pmax)


def test_dense_initial_model_estep_compact_os0_reuses_coarse_normalization_and_support(
    monkeypatch,
):
    calls = {}
    coarse_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 72, axis=0)
    coarse_translations = np.asarray([[0.0, 0.0], [1.5, -0.5]], dtype=np.float32)
    coarse_log_evidence = np.asarray([11.0, 12.0], dtype=np.float64)
    coarse_sum_weight = np.asarray([2.5e22, 3.0e22], dtype=np.float32)

    def fake_significance(dataset, *_args, **_kwargs):
        n_images = int(dataset.n_images)
        significant = [
            [np.asarray([class_idx], dtype=np.int32) for _ in range(n_images)]
            for class_idx in range(2)
        ]
        return (
            np.ones((2, 72), dtype=bool),
            np.full(n_images, 2, dtype=np.int32),
            np.zeros(n_images, dtype=np.int32),
            np.zeros(n_images, dtype=np.int32),
            significant,
            {
                "normalization_log_evidence": coarse_log_evidence,
                "relion_f32_sum_weight": coarse_sum_weight,
            },
        )

    def fake_run_compact(dataset, *_args, **kwargs):
        calls.update(kwargs["engine_kwargs"])
        return _fake_result(
            n_classes=2,
            n=8,
            n_images=int(dataset.n_units),
            n_groups=1,
        )

    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter._compute_k_class_significance_batched",
        fake_significance,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter._run_sparse_k_class_adaptive_pass2",
        fake_run_compact,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter._collapse_compact_pass2_rotation_stats_to_directions",
        lambda result, _n_psi: result,
    )
    monkeypatch.setattr(
        "recovar.em.initial_model.dense_adapter._restore_zero_oversampling_coarse_metadata",
        lambda result, **_kwargs: result,
    )

    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=2,
        nr_iter=1,
        n_directions=4,
        pseudo_halfsets=False,
    )
    config = DenseInitialModelEstepConfig(
        means=np.zeros((2, 8**3), dtype=np.complex64),
        mean_variance=np.ones((2, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=coarse_rotations,
        translations=coarse_translations,
        pass2_engine="compact",
        relion_bpref_frame=False,
        engine_kwargs={
            "sparse_pass2": True,
            "healpix_order": 0,
            "oversampling_order": 0,
            "coarse_translations": coarse_translations,
        },
    )

    result = run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([0, 1], dtype=np.int64),
    )

    assert calls["relion_fine_mstep_prune"] is False
    assert calls["relion_fine_mstep_keep_all"] is True
    np.testing.assert_array_equal(
        calls["relion_f32_normalization_sum_weight"],
        coarse_sum_weight,
    )
    assert "normalization_log_evidence" not in calls
    assert result.meta["pass2_engine"] == "compact"


def test_zero_oversampling_restores_k_class_coarse_argmax_metadata():
    from recovar.em.dense_single_volume.helpers.types import make_relion_stats
    from recovar.em.dense_single_volume.k_class import KClassEMResult
    from recovar.em.initial_model.dense_adapter import (
        _restore_zero_oversampling_coarse_metadata,
    )

    coarse_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 4, axis=0)
    coarse_rotations[:, 0, 0] = np.arange(4, dtype=np.float32)
    coarse_translations = np.asarray([[0.0, 0.0], [1.5, -0.5]], dtype=np.float32)
    coarse_hard = np.asarray([5, 6], dtype=np.int32)
    coarse_classes = np.asarray([1, 0], dtype=np.int32)
    coarse_pmax = np.asarray([0.25, 0.75], dtype=np.float32)
    fine_stats = make_relion_stats(
        log_evidence_per_image=np.asarray([1.0, 2.0], dtype=np.float32),
        best_log_score_per_image=np.asarray([0.5, 1.5], dtype=np.float32),
        max_posterior_per_image=np.asarray([0.9, 0.8], dtype=np.float32),
        rotation_posterior_sums=np.ones(4, dtype=np.float32),
    )
    fine_per_class_hard = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
    result = KClassEMResult(
        new_means=None,
        Ft_y=np.ones((2, 8**3), dtype=np.complex64),
        Ft_ctf=np.ones((2, 8**3), dtype=np.float32),
        per_class_hard_assignments=fine_per_class_hard,
        class_assignments=np.asarray([0, 1], dtype=np.int32),
        pose_assignments=np.asarray([0, 3], dtype=np.int32),
        class_responsibilities=np.full((2, 2), 0.5, dtype=np.float32),
        class_posterior_sums=np.ones(2, dtype=np.float32),
        stats=fine_stats,
        per_class_stats=(fine_stats, fine_stats),
        noise_stats=None,
        aggregate_noise_stats=None,
        best_pose_rotations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
        best_pose_translations=np.zeros((2, 2), dtype=np.float32),
        best_pose_rotation_ids=np.zeros(2, dtype=np.int32),
    )

    restored = _restore_zero_oversampling_coarse_metadata(
        result,
        hard_assignment=coarse_hard,
        class_assignment=coarse_classes,
        full_stats={
            "log_evidence_per_image": np.asarray([11.0, 12.0], dtype=np.float32),
            "best_log_score_per_image": np.asarray([10.0, 11.5], dtype=np.float32),
            "max_posterior_per_image": coarse_pmax,
        },
        coarse_rotations=coarse_rotations,
        coarse_translations=coarse_translations,
    )

    np.testing.assert_array_equal(restored.class_assignments, coarse_classes)
    np.testing.assert_array_equal(restored.pose_assignments, coarse_hard)
    np.testing.assert_array_equal(restored.per_class_hard_assignments, fine_per_class_hard)
    np.testing.assert_array_equal(restored.best_pose_rotation_ids, [2, 3])
    np.testing.assert_array_equal(restored.best_pose_rotations, coarse_rotations[[2, 3]])
    np.testing.assert_array_equal(restored.best_pose_translations, coarse_translations[[1, 0]])
    np.testing.assert_array_equal(restored.stats.max_posterior_per_image, coarse_pmax)
    np.testing.assert_array_equal(
        restored.per_class_stats[0].max_posterior_per_image,
        fine_stats.max_posterior_per_image,
    )


def test_dense_initial_model_estep_sparse_pass2_preserves_k_class_state(monkeypatch):
    calls = {"layouts": []}

    def fake_significance(dataset, means, noise_variance, rotations, translations, disc_type, **kwargs):
        del noise_variance, translations, disc_type
        calls["pass1_means_shape"] = np.asarray(means).shape
        calls["pass1_class_log_priors"] = np.asarray(kwargs["class_log_priors"], dtype=np.float64).copy()
        calls["pass1_relion_coarse_gaussian_default"] = kwargs[
            "relion_coarse_gaussian_default"
        ]
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
        calls["layouts"].append(
            {
                "significant_samples": [
                    np.asarray(samples, dtype=np.int32).copy() for samples in significant_samples
                ],
                "pass2_parent_prior": np.asarray(kwargs["translation_log_prior"], dtype=np.float32).copy(),
                "fine_prior": kwargs["fine_translation_log_prior"],
                "rotation_index_order": kwargs["rotation_index_order"],
                "allow_empty": kwargs["allow_empty"],
            }
        )
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
        del noise_variance, disc_type
        calls["pass2_means_shape"] = np.asarray(means).shape
        calls["pass2_mean_variance_shape"] = np.asarray(mean_variance).shape
        calls["pass2_class_log_priors"] = np.asarray(kwargs["class_log_priors"], dtype=np.float64).copy()
        calls["local_layout_count"] = len(local_layout)
        calls["has_class_local_rotation_log_prior"] = "class_local_rotation_log_prior" in kwargs
        calls["local_stats_use_reconstruction_probs"] = kwargs["stats_use_reconstruction_probs"]
        calls["local_class_posterior_sums_from_noise"] = kwargs["class_posterior_sums_from_noise"]
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
        pass2_engine="local",
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
    assert calls["pass1_relion_coarse_gaussian_default"] is False
    assert calls["local_layout_count"] == 2
    assert calls["has_class_local_rotation_log_prior"] is False
    assert calls["local_stats_use_reconstruction_probs"] is True
    assert calls["local_class_posterior_sums_from_noise"] is False
    assert len(calls["layouts"]) == 2
    for layout_call in calls["layouts"]:
        np.testing.assert_allclose(layout_call["pass2_parent_prior"], fine_prior[[1, 3]])
        assert layout_call["fine_prior"] is None
        assert layout_call["rotation_index_order"] == "relion_hidden"
        assert layout_call["allow_empty"] is True
    np.testing.assert_array_equal(calls["layouts"][0]["significant_samples"][0], [0])
    np.testing.assert_array_equal(calls["layouts"][0]["significant_samples"][1], [0])
    np.testing.assert_array_equal(calls["layouts"][1]["significant_samples"][0], [1])
    np.testing.assert_array_equal(calls["layouts"][1]["significant_samples"][1], [1])
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


def test_sparse_pass2_pass1_current_size_uses_pre_update_healpix_order():
    """InitialModel sizes pass 1 before RELION promotes the sampling order."""
    state = initialise_denovo_state(
        ori_size=128,
        pixel_size=4.25,
        K=1,
        nr_iter=25,
        n_directions=192,
        pseudo_halfsets=False,
    )
    state.current_size = 56

    pass1_current_size = _resolve_sparse_pass1_current_size(
        state,
        {"current_size": state.current_size},
        {
            "healpix_order": 2,
            "pass1_healpix_order": 1,
            "particle_diameter_ang": 200.0,
        },
    )

    assert pass1_current_size == 26


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
