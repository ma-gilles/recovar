"""Focused guards for the opt-in K=4 InitialModel native BPref replay."""

from types import SimpleNamespace

import numpy as np
import pytest

import recovar.em.dense_single_volume.k_class as k_class
from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed
from recovar.em.initial_model import driver

pytestmark = pytest.mark.unit


def _rotations(count: int) -> np.ndarray:
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (count, 3, 3)).copy()
    rotations[:, 0, 0] = np.arange(1, count + 1, dtype=np.float32)
    return rotations


def _valid_gate_kwargs(**overrides):
    kwargs = {
        "requested": True,
        "n_classes": 4,
        "initial_model_iteration": 1,
        "use_relion_x_half_mstep": True,
        "use_exact_relion_gaussian": True,
        "use_float64_scoring": False,
        "device_signature_requested": False,
        "current_size": 20,
    }
    kwargs.update(overrides)
    return kwargs


def test_k4_native_bpref_gate_is_opt_in_fail_closed_and_k1_dormant():
    assert (
        driver.NativeInitialModelOptions(
            fn_img="particles.star"
        ).relion_kclass_firstiter_native_bpref_replay
        is False
    )
    assert bucketed._kclass_firstiter_native_bpref_replay_enabled(
        **_valid_gate_kwargs(requested=False, n_classes=1)
    ) is False
    assert bucketed._kclass_firstiter_native_bpref_replay_enabled(
        **_valid_gate_kwargs()
    ) is True

    for invalid in (
        {"n_classes": 1},
        {"initial_model_iteration": 2},
        {"use_relion_x_half_mstep": False},
        {"use_exact_relion_gaussian": False},
        {"use_float64_scoring": True},
        {"device_signature_requested": True},
        {"current_size": None},
    ):
        with pytest.raises(ValueError, match="K-class firstiter native BPref replay"):
            bucketed._kclass_firstiter_native_bpref_replay_enabled(
                **_valid_gate_kwargs(**invalid)
            )


def test_k4_native_bpref_option_enters_initial_model_engine_kwargs_only_when_set():
    plan = driver.NativeSamplingPlan(
        rotations=np.eye(3, dtype=np.float32)[None, :, :],
        translations=np.zeros((1, 2), dtype=np.float32),
        random_perturbation=0.0,
        oversampling=0,
    )
    dataset = SimpleNamespace(voxel_size=1.0)
    common = (
        dataset,
        np.ones(4, dtype=np.float32),
        plan,
        np.zeros((1, 2), dtype=np.float32),
    )
    disabled = driver._dense_estep_config(
        common[0],
        driver.NativeInitialModelOptions(fn_img="particles.star"),
        *common[1:],
    )
    enabled = driver._dense_estep_config(
        common[0],
        driver.NativeInitialModelOptions(
            fn_img="particles.star",
            relion_kclass_firstiter_native_bpref_replay=True,
        ),
        *common[1:],
    )

    assert "relion_kclass_firstiter_native_bpref_replay" not in disabled.engine_kwargs
    assert enabled.engine_kwargs["relion_kclass_firstiter_native_bpref_replay"] is True


def test_k4_native_bpref_raw_operands_are_staged_once_bitwise():
    images_a = np.asarray(
        [[1 + 2j, 3 + 4j, 5 + 6j], [7 + 8j, 9 + 10j, 11 + 12j]],
        dtype=np.complex64,
    )
    ctf_a = np.asarray([[0.25, -0.5, 0.75], [1.0, 1.25, -1.5]], dtype=np.float32)
    noise = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)
    store = bucketed._stage_kclass_native_bpref_raw_operands(
        None,
        raw_images=images_a,
        raw_ctf=ctf_a,
        raw_minvsigma2=noise,
        particle_half_local_indices=np.asarray([2, 0]),
        particle_original_indices=np.asarray([30, 10]),
    )
    images_b = np.asarray([[13 + 14j, 15 + 16j, 17 + 18j]], dtype=np.complex64)
    ctf_b = np.asarray([[1.75, -2.0, 2.25]], dtype=np.float32)
    store = bucketed._stage_kclass_native_bpref_raw_operands(
        store,
        raw_images=images_b,
        raw_ctf=ctf_b,
        raw_minvsigma2=noise.copy(),
        particle_half_local_indices=np.asarray([1]),
        particle_original_indices=np.asarray([20]),
    )

    np.testing.assert_array_equal(store.raw_minvsigma2, noise)
    np.testing.assert_array_equal(store.by_half_local_index[2].raw_image, images_a[0])
    np.testing.assert_array_equal(store.by_half_local_index[0].raw_ctf, ctf_a[1])
    np.testing.assert_array_equal(store.by_half_local_index[1].raw_image, images_b[0])
    assert store.raw_minvsigma2.dtype == noise.dtype
    assert store.by_half_local_index[2].raw_image.dtype == images_a.dtype
    with pytest.raises(MemoryError, match="host staging exceeded"):
        bucketed._check_kclass_native_bpref_host_limit(store, [], 1)

    with pytest.raises(ValueError, match="more than once"):
        bucketed._stage_kclass_native_bpref_raw_operands(
            store,
            raw_images=images_b,
            raw_ctf=ctf_b,
            raw_minvsigma2=noise,
            particle_half_local_indices=np.asarray([1]),
            particle_original_indices=np.asarray([20]),
        )


def test_k4_native_bpref_raw_operand_builder_preserves_native_units(monkeypatch):
    source_batch = np.arange(8, dtype=np.float32).reshape(2, 4)
    raw_half = np.asarray(
        [
            [1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j],
            [9 + 10j, 11 + 12j, 13 + 14j, 15 + 16j],
        ],
        dtype=np.complex64,
    )
    source_ctf = np.asarray(
        [[0.125, -0.25, 0.5, 1.0], [1.25, -1.5, 1.75, -2.0]],
        dtype=np.float64,
    )
    batch_scale = np.asarray([2.0, 0.5], dtype=np.float32)
    calls = []

    def fake_prepare(*args, **kwargs):
        calls.append(("prepare", args[1] is source_batch, kwargs))
        return True, None, np.ones(2, dtype=np.float32), batch_scale, {"sentinel": 7}

    def fake_process(_dataset, batch, masked, *, relion_preprocess_kwargs):
        calls.append(
            (
                "process",
                batch is source_batch,
                masked,
                dict(relion_preprocess_kwargs),
            )
        )
        return raw_half

    monkeypatch.setattr(bucketed, "prepare_batch_preprocess_operands", fake_prepare)
    monkeypatch.setattr(bucketed, "process_half_image", fake_process)
    monkeypatch.setattr(
        bucketed,
        "_relion_exact_ctf_half_from_source_star",
        lambda *_args, **_kwargs: source_ctf,
    )
    noise = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    image, ctf, minvsigma2 = bucketed._prepare_relion_native_bpref_raw_operands(
        object(),
        source_batch,
        np.asarray([4, 9], dtype=np.int64),
        noise,
        image_corrections=np.ones(2, dtype=np.float32),
        scale_corrections=batch_scale,
        image_pre_shifts=None,
        image_shape=(2, 2),
    )

    expected_image = (
        raw_half * batch_scale[:, None] * np.float32(1.0 / 4.0)
    ).astype(np.complex64)
    expected_noise = np.reciprocal(
        noise.astype(np.float64) / np.float64(16.0)
    ).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(image), expected_image)
    np.testing.assert_array_equal(np.asarray(ctf), source_ctf.astype(np.float32))
    np.testing.assert_array_equal(np.asarray(minvsigma2), expected_noise)
    assert np.asarray(image).dtype == np.complex64
    assert np.asarray(ctf).dtype == np.float32
    assert np.asarray(minvsigma2).dtype == np.float32
    assert calls[0][0:2] == ("prepare", True)
    assert calls[1] == (
        "process",
        True,
        False,
        {"sentinel": 7, "relion_fft_per_image": True},
    )


def test_k4_native_bpref_rectangular_staging_removes_rotation_padding():
    posterior = np.zeros((4, 3), dtype=np.float32)
    posterior[:2] = np.asarray([[0.1, 0.2, 0.0], [0.3, 0.0, 0.4]], dtype=np.float32)
    contribution = bucketed._stage_kclass_native_bpref_contribution(
        class_index=2,
        half_local_index=5,
        original_index=17,
        posterior=posterior,
        rotations=_rotations(4),
        actual_rotation_count=2,
        n_translations=3,
    )

    assert contribution.rotations.shape == (2, 3, 3)
    assert contribution.dense_posterior.shape == (2, 3)
    np.testing.assert_array_equal(
        bucketed._materialize_kclass_native_bpref_posterior(contribution),
        posterior[:2],
    )

    posterior[3, 1] = np.float32(1e-3)
    with pytest.raises(ValueError, match="padding carries posterior mass"):
        bucketed._stage_kclass_native_bpref_contribution(
            class_index=2,
            half_local_index=5,
            original_index=17,
            posterior=posterior,
            rotations=_rotations(4),
            actual_rotation_count=2,
            n_translations=3,
        )


def test_k4_native_bpref_compact_staging_retains_only_sparse_valid_slots():
    pair_posterior = np.asarray([0.2, 0.0, 0.3, 0.4, 0.0], dtype=np.float32)
    pair_rotation_row = np.asarray([1, 99, 0, 1, -1], dtype=np.int32)
    pair_translation_index = np.asarray([2, 99, 0, 1, -1], dtype=np.int32)
    pair_mask = np.asarray([True, False, True, True, False])
    contribution = bucketed._stage_kclass_native_bpref_contribution(
        class_index=1,
        half_local_index=3,
        original_index=8,
        posterior=pair_posterior,
        rotations=_rotations(4),
        actual_rotation_count=2,
        n_translations=3,
        pair_rotation_row=pair_rotation_row,
        pair_translation_index=pair_translation_index,
        pair_mask=pair_mask,
    )

    assert contribution.rotations.shape == (2, 3, 3)
    assert contribution.pair_posterior.shape == (3,)
    expected = np.asarray([[0.3, 0.0, 0.0], [0.0, 0.4, 0.2]], dtype=np.float32)
    np.testing.assert_array_equal(
        bucketed._materialize_kclass_native_bpref_posterior(contribution),
        expected,
    )

    pair_posterior[1] = np.float32(1e-4)
    with pytest.raises(ValueError, match="padding carries posterior mass"):
        bucketed._stage_kclass_native_bpref_contribution(
            class_index=1,
            half_local_index=3,
            original_index=8,
            posterior=pair_posterior,
            rotations=_rotations(4),
            actual_rotation_count=2,
            n_translations=3,
            pair_rotation_row=pair_rotation_row,
            pair_translation_index=pair_translation_index,
            pair_mask=pair_mask,
        )


def test_k4_native_bpref_compact_and_rectangular_posteriors_are_bitwise_equal():
    dense = np.asarray([[0.05, 0.15, 0.0], [0.2, 0.25, 0.35]], dtype=np.float32)
    rectangular = bucketed._stage_kclass_native_bpref_contribution(
        class_index=0,
        half_local_index=0,
        original_index=4,
        posterior=dense,
        rotations=_rotations(2),
        actual_rotation_count=2,
        n_translations=3,
    )
    compact = bucketed._stage_kclass_native_bpref_contribution(
        class_index=0,
        half_local_index=0,
        original_index=4,
        posterior=np.asarray([0.35, 0.05, 0.2, 0.25, 0.15], dtype=np.float32),
        rotations=_rotations(2),
        actual_rotation_count=2,
        n_translations=3,
        pair_rotation_row=np.asarray([1, 0, 1, 1, 0]),
        pair_translation_index=np.asarray([2, 0, 0, 1, 1]),
        pair_mask=np.ones(5, dtype=bool),
    )

    np.testing.assert_array_equal(
        bucketed._materialize_kclass_native_bpref_posterior(compact),
        bucketed._materialize_kclass_native_bpref_posterior(rectangular),
    )
    np.testing.assert_array_equal(compact.rotations, rectangular.rotations)


def test_k4_native_bpref_orders_mixed_buckets_and_preserves_soft_class_mass():
    images = np.asarray([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex64)
    ctf = np.asarray([[0.5, 0.75], [1.0, 1.25]], dtype=np.float32)
    store = bucketed._stage_kclass_native_bpref_raw_operands(
        None,
        raw_images=images,
        raw_ctf=ctf,
        raw_minvsigma2=np.asarray([2.0, 4.0], dtype=np.float32),
        particle_half_local_indices=np.asarray([1, 0]),
        particle_original_indices=np.asarray([30, 10]),
    )
    class_masses = np.asarray(
        [[0.10, 0.20], [0.20, 0.30], [0.30, 0.10], [0.40, 0.40]],
        dtype=np.float32,
    )
    contributions = []
    for class_index in (3, 1, 0, 2):
        for half_local_index, original_index, particle_column in ((1, 30, 1), (0, 10, 0)):
            contributions.append(
                bucketed._stage_kclass_native_bpref_contribution(
                    class_index=class_index,
                    half_local_index=half_local_index,
                    original_index=original_index,
                    posterior=np.asarray(
                        [[class_masses[class_index, particle_column], 0.0]],
                        dtype=np.float32,
                    ),
                    rotations=_rotations(1),
                    actual_rotation_count=1,
                    n_translations=2,
                )
            )

    ordered = bucketed._ordered_kclass_native_bpref_contributions(
        store,
        contributions,
        n_classes=4,
    )
    for class_rows in ordered:
        assert [row.original_index for row in class_rows] == [10, 30]
    recovered = np.asarray(
        [
            [
                np.sum(bucketed._materialize_kclass_native_bpref_posterior(row))
                for row in class_rows
            ]
            for class_rows in ordered
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(recovered, class_masses)
    np.testing.assert_array_equal(np.sum(recovered, axis=0), np.ones(2, dtype=np.float32))


class _CapturedFusedCall(RuntimeError):
    pass


@pytest.mark.parametrize("n_classes", [1, 4])
def test_opt_in_off_does_not_forward_replay_keywords_to_existing_paths(
    monkeypatch,
    n_classes,
):
    captured = {}

    def capture_fused(*args, **kwargs):
        del args
        captured.update(kwargs)
        raise _CapturedFusedCall

    monkeypatch.setattr(k_class, "_use_fused_sparse_k_class_pass2", lambda _k: True)
    monkeypatch.setattr(
        bucketed,
        "compute_k_class_pass2_stats_sparse_fused",
        capture_fused,
    )
    with pytest.raises(_CapturedFusedCall):
        k_class._run_sparse_k_class_adaptive_pass2(
            object(),
            np.zeros((n_classes, 1), dtype=np.complex64),
            np.ones(1, dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.eye(3, dtype=np.float32)[None, :, :],
            np.zeros((1, 2), dtype=np.float32),
            np.eye(3, dtype=np.float32)[None, :, :],
            None,
            np.zeros(1, dtype=np.int64),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            [[np.asarray([0], dtype=np.int32)] for _ in range(n_classes)],
            "linear_interp",
            class_log_priors=np.log(
                np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
            ),
            accumulate_noise=False,
            return_best_pose_details=False,
            coarse_healpix_order=0,
            oversampling_order=0,
            random_perturbation=0.0,
            engine_kwargs={"relion_exact_fine_gaussian": False},
        )

    assert "relion_kclass_firstiter_native_bpref_replay" not in captured
    assert "initial_model_iteration" not in captured


def test_k4_native_bpref_replay_is_class_major_ordered_and_single_triplet(
    monkeypatch,
):
    images = np.asarray(
        [[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]],
        dtype=np.complex64,
    )
    store = bucketed._stage_kclass_native_bpref_raw_operands(
        None,
        raw_images=images,
        raw_ctf=np.ones((2, 2), dtype=np.float32),
        raw_minvsigma2=np.ones(2, dtype=np.float32),
        particle_half_local_indices=np.asarray([1, 0]),
        particle_original_indices=np.asarray([30, 10]),
    )
    class_masses = np.asarray(
        [
            [0.125, 0.250],
            [0.250, 0.125],
            [0.375, 0.375],
            [0.250, 0.250],
        ],
        dtype=np.float32,
    )
    contributions = []
    for class_index in (2, 0, 3, 1):
        for local_index, original_index, particle_column in ((1, 30, 1), (0, 10, 0)):
            mass = class_masses[class_index, particle_column]
            contributions.append(
                bucketed._stage_kclass_native_bpref_contribution(
                    class_index=class_index,
                    half_local_index=local_index,
                    original_index=original_index,
                    posterior=np.asarray([[mass / 2, mass / 2]], dtype=np.float32),
                    rotations=_rotations(1),
                    actual_rotation_count=1,
                    n_translations=2,
                )
            )

    events = []
    active_triplets = 0
    zero_count = 0
    peak_triplets = 0

    def tracked_zeros(size, *, dtype):
        nonlocal active_triplets, peak_triplets, zero_count
        zero_count += 1
        if zero_count % 3 == 0:
            active_triplets += 1
            peak_triplets = max(peak_triplets, active_triplets)
            events.append(("triplet", active_triplets))
        return np.zeros(size, dtype=np.dtype(dtype))

    def fake_accumulate(
        _images,
        _ctf,
        _noise,
        posterior,
        _rotations_arg,
        _counts,
        _local_indices,
        original_indices,
        data_real,
        data_imag,
        weight,
        **_kwargs,
    ):
        mass = np.float32(np.sum(np.asarray(posterior), dtype=np.float32))
        original_index = int(np.asarray(original_indices)[0])
        events.append(("launch", original_index, float(mass)))
        next_real = np.array(data_real, copy=True)
        next_imag = np.array(data_imag, copy=True)
        next_weight = np.array(weight, copy=True)
        next_real[0] += mass
        next_imag[0] += mass * np.float32(original_index)
        next_weight[0] += np.float32(2.0) * mass
        return next_real, next_imag, next_weight

    def fake_finalize(data_real, data_imag, weight, *_args, **_kwargs):
        nonlocal active_triplets
        assert active_triplets == 1
        active_triplets -= 1
        events.append(("finalize",))
        return data_real + np.complex64(1j) * data_imag, weight

    monkeypatch.setattr(bucketed.jnp, "zeros", tracked_zeros)
    monkeypatch.setattr(
        bucketed,
        "_accumulate_relion_firstiter_bpref_fused_split",
        fake_accumulate,
    )
    monkeypatch.setattr(
        bucketed,
        "_normalize_split_relion_firstiter_bpref_accumulators",
        lambda real, imag, weight, *_args: (real, imag, weight),
    )
    monkeypatch.setattr(bucketed, "finalize_split_relion_x_half_bpref", fake_finalize)
    monkeypatch.setattr(
        bucketed,
        "relion_x_half_accumulators_to_public_layout",
        lambda data, weight, _shape: (data, weight),
    )
    monkeypatch.setattr(bucketed, "_maybe_dump_split_native_half_mstep", lambda *_a, **_k: None)
    monkeypatch.setattr(bucketed, "_maybe_dump_native_half_mstep", lambda *_a, **_k: None)
    monkeypatch.setattr(bucketed.jax, "block_until_ready", lambda value: value)

    Ft_y, Ft_ctf = bucketed._replay_kclass_firstiter_native_bpref(
        store,
        contributions,
        n_classes=4,
        recon_volume_size=2,
        centered_pixel_indices=np.asarray([0, 1], dtype=np.int32),
        fftw_pixel_indices=np.asarray([0, 1], dtype=np.int32),
        translation_angles=np.zeros((2, 2), dtype=np.float32),
        physical_image_shape=(2, 2),
        volume_shape=(2, 2, 2),
        max_r=1.0,
        adaptive_fraction=0.999,
        current_size=2,
        n_images=2,
        symmetry_label="C1",
    )

    assert peak_triplets == 1
    assert active_triplets == 0
    assert events == [
        ("triplet", 1),
        ("launch", 10, 0.125),
        ("launch", 30, 0.25),
        ("finalize",),
        ("triplet", 1),
        ("launch", 10, 0.25),
        ("launch", 30, 0.125),
        ("finalize",),
        ("triplet", 1),
        ("launch", 10, 0.375),
        ("launch", 30, 0.375),
        ("finalize",),
        ("triplet", 1),
        ("launch", 10, 0.25),
        ("launch", 30, 0.25),
        ("finalize",),
    ]
    expected_mass = np.sum(class_masses, axis=1, dtype=np.float32)
    expected_index_weighted = (
        class_masses[:, 0] * np.float32(10.0)
        + class_masses[:, 1] * np.float32(30.0)
    )
    for class_index in range(4):
        np.testing.assert_array_equal(
            Ft_y[class_index],
            np.asarray(
                [
                    expected_mass[class_index]
                    + np.complex64(1j) * expected_index_weighted[class_index],
                    0.0,
                ],
                dtype=np.complex64,
            ),
        )
        np.testing.assert_array_equal(
            Ft_ctf[class_index],
            np.asarray([2.0 * expected_mass[class_index], 0.0], dtype=np.float32),
        )
