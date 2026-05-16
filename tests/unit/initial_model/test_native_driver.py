"""Native InitialModel driver tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import recovar.em.initial_model.driver as driver
from recovar.data_io.starfile import read_star
from recovar.em.initial_model import initialise_denovo_state
from recovar.em.initial_model.dense_adapter import DenseInitialModelEstepResult
from recovar.em.initial_model.m_step import VdamAccumulator

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_ab_initio.py"

pytestmark = pytest.mark.unit


def _load_run_ab_initio():
    import sys

    spec = importlib.util.spec_from_file_location("run_ab_initio_native_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_ab_initio_native_test"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("run_ab_initio_native_test", None)
        raise
    return module


def test_micrograph_sort_order_matches_relion_experiment_order():
    main = pd.DataFrame(
        {
            "_rlnMicrographName": ["1", "2", "10", "100", "11"],
            "_rlnImageName": ["1@s.mrcs", "2@s.mrcs", "3@s.mrcs", "4@s.mrcs", "5@s.mrcs"],
        }
    )

    assert driver._micrograph_sort_order(main).tolist() == [0, 2, 3, 4, 1]


def test_experiment_read_order_uses_micrograph_lexicographic_order():
    main = pd.DataFrame(
        {
            "_rlnMicrographName": ["1", "2", "10", "100", "11"],
            "_rlnImageName": ["1@s.mrcs", "2@s.mrcs", "3@s.mrcs", "4@s.mrcs", "5@s.mrcs"],
        }
    )

    assert driver._experiment_read_order(main).tolist() == [0, 2, 3, 4, 1]


def test_translation_log_prior_matches_relion_pdf_offset_scaling():
    translations = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, -1.0]], dtype=np.float32)

    prior = driver._translation_log_prior(translations, voxel_size=3.0, sigma_angstrom=6.0)

    np.testing.assert_allclose(prior, np.asarray([0.0, -4.5, -1.125], dtype=np.float32), rtol=1e-6)

    centered = driver._translation_log_prior(
        translations,
        voxel_size=3.0,
        sigma_angstrom=6.0,
        centers=np.asarray([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        centered,
        np.asarray([[-1.125, -10.125, -2.25], [-1.125, -5.625, -4.5]], dtype=np.float32),
        rtol=1e-6,
    )


def test_image_pre_shifts_from_star_converts_angstrom_origins_to_rounded_pixels():
    main = pd.DataFrame(
        {
            "_rlnOriginXAngst": ["4.2", "-3.9", "0.0"],
            "_rlnOriginYAngst": ["-8.1", "2.0", "0.4"],
        }
    )

    raw = driver._image_origin_offsets_pixels_from_star(main, SimpleNamespace(voxel_size=2.0))
    shifts = driver._image_pre_shifts_from_star(main, SimpleNamespace(voxel_size=2.0))

    np.testing.assert_allclose(
        raw,
        np.asarray([[2.1, -4.05], [-1.95, 1.0], [0.0, 0.2]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        shifts,
        np.asarray([[2.0, -4.0], [-2.0, 1.0], [0.0, 0.0]], dtype=np.float32),
    )


def test_image_pre_shifts_from_star_uses_legacy_pixel_origins():
    main = pd.DataFrame({"_rlnOriginX": ["0.5", "-1.4"], "_rlnOriginY": ["1.6", "-0.49"]})

    shifts = driver._image_pre_shifts_from_star(main, SimpleNamespace(voxel_size=2.0))

    np.testing.assert_array_equal(shifts, np.asarray([[0.0, 2.0], [-1.0, 0.0]], dtype=np.float32))


def test_image_pre_shifts_from_star_defaults_to_zero_without_origins():
    main = pd.DataFrame({"_rlnImageName": ["1@stack.mrcs", "2@stack.mrcs"]})

    shifts = driver._image_pre_shifts_from_star(main, SimpleNamespace(voxel_size=2.0))

    np.testing.assert_array_equal(shifts, np.zeros((2, 2), dtype=np.float32))


def test_particle_state_from_star_preserves_class_and_pmax_columns():
    main = pd.DataFrame(
        {
            "_rlnImageName": ["1@stack.mrcs", "2@stack.mrcs"],
            "_rlnClassNumber": ["2", "1"],
            "_rlnMaxValueProbDistribution": ["0.9", "0.25"],
        }
    )

    state = driver._particle_state_from_star(main, SimpleNamespace(voxel_size=1.0, n_images=2))

    np.testing.assert_array_equal(state.translation_offsets, np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(state.class_assignments, [1, 0])
    np.testing.assert_allclose(state.max_posterior, [0.9, 0.25])
    np.testing.assert_array_equal(state.pose_assignments, [-1, -1])


def test_sampling_plan_oversamples_relion_grid():
    opts = driver.NativeInitialModelOptions(
        fn_img="particles.star",
        healpix_order=1,
        oversampling=1,
        random_perturbation=0.0,
    )

    plan = driver._build_sampling_plan(opts)

    assert plan.rotations.shape == (4608, 3, 3)
    assert plan.translations.shape == (116, 2)
    assert plan.random_perturbation == 0.0


def test_initial_sampling_state_uses_relion_angstrom_internal_units():
    opts = driver.NativeInitialModelOptions(
        fn_img="particles.star",
        healpix_order=1,
        oversampling=1,
        offset_range_px=6.0,
        offset_step_px=2.0,
        random_perturbation=0.0,
    )

    sampling_state = driver._initial_sampling_state(opts, pixel_size=2.125)
    plan = driver._build_sampling_plan(opts, iteration=1, sampling_state=sampling_state)

    assert sampling_state.offset_range_angstrom == pytest.approx(12.75)
    assert sampling_state.offset_step_angstrom == pytest.approx(4.25)
    assert sampling_state.offset_range_px == pytest.approx(6.0)
    assert sampling_state.offset_step_px == pytest.approx(2.0)
    assert sampling_state.effective_offset_step_angstrom == pytest.approx(2.125)
    assert plan.offset_range_angstrom == pytest.approx(12.75)
    assert plan.offset_step_angstrom == pytest.approx(4.25)
    assert plan.rotations.shape == (4608, 3, 3)
    assert plan.translations.shape == (116, 2)


def test_native_sampling_updates_like_relion_gradient_initialmodel_default():
    opts = driver.NativeInitialModelOptions(
        fn_img="particles.star",
        nr_iter=200,
        healpix_order=1,
        oversampling=1,
        offset_range_px=6.0,
        offset_step_px=2.0,
    )
    sampling_state = driver._initial_sampling_state(opts, pixel_size=2.125)
    state = initialise_denovo_state(ori_size=256, pixel_size=2.125, K=1, nr_iter=200, n_directions=1)
    state.current_resolution = 1.0 / 108.8

    assert driver._prepare_native_sampling_for_iteration(
        sampling_state,
        state,
        iteration=9,
        do_grad=True,
    ) is False
    assert sampling_state.healpix_order == 1
    assert sampling_state.offset_range_angstrom == pytest.approx(12.75)
    assert sampling_state.offset_step_angstrom == pytest.approx(4.25)

    assert driver._prepare_native_sampling_for_iteration(
        sampling_state,
        state,
        iteration=10,
        do_grad=True,
    ) is True
    assert sampling_state.healpix_order == 2
    assert sampling_state.offset_range_angstrom == pytest.approx(8.2875)
    assert sampling_state.offset_step_angstrom == pytest.approx(3.0)
    assert sampling_state.effective_offset_step_angstrom == pytest.approx(1.5)

    sampling_state.current_changes_optimal_offsets_angstrom = 2.614243
    assert driver._prepare_native_sampling_for_iteration(
        sampling_state,
        state,
        iteration=20,
        do_grad=True,
    ) is True
    assert sampling_state.healpix_order == 3
    assert sampling_state.offset_range_angstrom == pytest.approx(10.77375)
    assert sampling_state.offset_step_angstrom == pytest.approx(3.0)

    sampling_state.current_changes_optimal_offsets_angstrom = 2.0
    assert driver._prepare_native_sampling_for_iteration(
        sampling_state,
        state,
        iteration=30,
        do_grad=True,
    ) is True
    assert sampling_state.healpix_order == 3
    assert sampling_state.offset_range_angstrom == pytest.approx(10.0)
    assert sampling_state.offset_step_angstrom == pytest.approx(3.0)


def test_active_relion_initialmodel_max_significants_matches_gradient_default():
    state = initialise_denovo_state(ori_size=16, pixel_size=1.0, K=3, nr_iter=8, n_directions=1)

    assert driver._active_relion_initialmodel_max_significants(state, do_grad=True) == 300
    assert driver._active_relion_initialmodel_max_significants(state, do_grad=False) == -1


def test_effective_class_support_floor_zeros_subparticle_accumulators():
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=4, nr_iter=4, n_directions=1)

    accumulators = []
    for halfset_idx in (0, 1):
        for class_idx in range(4):
            accumulators.append(
                VdamAccumulator(
                    data=np.full((3, 3, 2), class_idx + 1, dtype=np.complex128),
                    weight=np.full((3, 3, 2), class_idx + 1, dtype=np.float64),
                    class_idx=class_idx,
                    halfset_idx=halfset_idx,
                )
            )
    result = DenseInitialModelEstepResult(
        accumulators=accumulators,
        meta={
            "selected_particle_ids": np.asarray([13, 10, 12, 11], dtype=np.int64),
            "class_assignments": np.zeros(4, dtype=np.int32),
            "class_posterior_sums": np.asarray([12.0, 4.0, 3.0, 0.0], dtype=np.float64),
            "halfset_0_class_posterior_sums": np.asarray([6.0, 1.5, 1.0, 0.0], dtype=np.float64),
            "halfset_1_class_posterior_sums": np.asarray([6.0, 2.5, 2.0, 0.0], dtype=np.float64),
            "class_reconstruction_support_sums": np.asarray([12.0, 0.0017, 3.0, 0.0], dtype=np.float64),
            "halfset_0_class_reconstruction_support_sums": np.asarray([6.0, 0.0017, 1.0, 0.0], dtype=np.float64),
            "halfset_1_class_reconstruction_support_sums": np.asarray([6.0, 0.0, 2.0, 0.0], dtype=np.float64),
            "class_bpref_weight_sums": np.asarray([90.0, 8.0, 30.0, 0.0], dtype=np.float64),
            "class_direction_posterior_sums": np.ones((4, 2), dtype=np.float64),
        },
        halfset_results={},
    )

    masked = driver._apply_effective_class_support_floor(result, state)

    np.testing.assert_array_equal(masked.meta["class_effective_support_active"], [True, False, True, False])
    np.testing.assert_allclose(masked.meta["class_reconstruction_support_sums_raw"], [12.0, 0.0017, 3.0, 0.0])
    np.testing.assert_allclose(masked.meta["class_posterior_sums_raw"], [12.0, 4.0, 3.0, 0.0])
    np.testing.assert_allclose(masked.meta["class_bpref_weight_sums_raw"], [90.0, 8.0, 30.0, 0.0])
    np.testing.assert_allclose(masked.meta["class_bpref_weight_sums"], [90.0, 0.0, 30.0, 0.0])
    np.testing.assert_allclose(masked.meta["class_posterior_sums"], [11.25, 0.0, 3.75, 0.0])
    np.testing.assert_allclose(masked.meta["class_reconstruction_support_sums"], [12.0, 0.0, 3.0, 0.0])
    np.testing.assert_allclose(masked.meta["halfset_0_class_posterior_sums_raw"], [6.0, 1.5, 1.0, 0.0])
    np.testing.assert_allclose(masked.meta["halfset_0_class_posterior_sums"], [6.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(masked.meta["halfset_1_class_posterior_sums"], [6.0, 0.0, 2.0, 0.0])
    np.testing.assert_allclose(masked.meta["halfset_0_class_reconstruction_support_sums"], [6.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(masked.meta["halfset_1_class_reconstruction_support_sums"], [6.0, 0.0, 2.0, 0.0])
    np.testing.assert_allclose(masked.meta["class_direction_posterior_sums"][[1, 3]], 0.0)
    for accum in masked.accumulators:
        if accum.class_idx in (1, 3):
            assert not np.any(accum.weight)
            assert not np.any(accum.data)
        else:
            assert np.any(accum.weight)


def test_random_perturbation_override_is_fixed():
    opts = driver.NativeInitialModelOptions(
        fn_img="particles.star",
        random_perturbation=-0.125,
    )

    assert driver._random_perturbation_for_iteration(opts, 1) == -0.125
    assert driver._random_perturbation_for_iteration(opts, 7) == -0.125


def test_random_perturbation_sequence_matches_relion_initialmodel_fixture():
    opts = driver.NativeInitialModelOptions(
        fn_img="particles.star",
        random_seed=1776701668,
        perturbation_factor=0.5,
    )

    assert driver._random_perturbation_for_iteration(opts, 1) == pytest.approx(-0.25278, abs=5e-6)
    assert driver._random_perturbation_for_iteration(opts, 2) == pytest.approx(0.125066, abs=5e-6)


def test_initial_state_applies_relion_bootstrap_postprocess(monkeypatch):
    monkeypatch.delenv("RECOVAR_INITIAL_IREF_OVERRIDE", raising=False)
    raw_iref = np.full((1, 8, 8, 8), 2.0, dtype=np.float64)
    post_iref = np.full((1, 8, 8, 8), 3.0, dtype=np.float64)
    calls = []

    def fake_avg(*args, **kwargs):
        return np.zeros((8, 8), dtype=np.float64), np.ones((1, 5), dtype=np.float64)

    def fake_load_raw_images(dataset, particle_ids, *, batch_size):
        np.testing.assert_array_equal(particle_ids, np.asarray([1, 0], dtype=np.int64))
        return np.zeros((2, 8, 8), dtype=np.float64)

    def fake_bootstrap(**kwargs):
        assert kwargs["ori_size"] == 8
        assert kwargs["nr_classes"] == 1
        assert kwargs["particle_diameter_ang"] == 16.0
        assert kwargs.get("particle_seed_ids") is None
        return raw_iref.copy()

    def fake_postprocess(iref, **kwargs):
        np.testing.assert_array_equal(iref, raw_iref)
        calls.append(kwargs)
        return post_iref.copy()

    monkeypatch.setattr(driver, "compute_avg_unaligned_and_sigma2", fake_avg)
    monkeypatch.setattr(driver, "_load_raw_images", fake_load_raw_images)
    monkeypatch.setattr(driver, "compute_bootstrap_iref_via_cpp", fake_bootstrap)
    monkeypatch.setattr(driver, "postprocess_bootstrap_iref_via_cpp", fake_postprocess)

    main = pd.DataFrame(
        {
            "_rlnImageName": ["1@stack.mrcs", "2@stack.mrcs"],
            "_rlnMicrographName": ["b", "a"],
            "_rlnOpticsGroup": ["1", "1"],
            "_rlnDefocusU": ["10000", "10000"],
            "_rlnDefocusV": ["10000", "10000"],
            "_rlnDefocusAngle": ["0", "0"],
        }
    )
    optics = pd.DataFrame(
        {
            "_rlnVoltage": ["300"],
            "_rlnSphericalAberration": ["2.7"],
            "_rlnAmplitudeContrast": ["0.07"],
        }
    )
    dataset = SimpleNamespace(grid_size=8, voxel_size=2.0, n_images=2)
    opts = driver.NativeInitialModelOptions(
        fn_img="particles.star",
        nr_classes=1,
        nr_iter=1,
        particle_diameter=16.0,
        image_batch_size=2,
        bootstrap_min_particles=2,
    )

    state, optics_groups = driver._initial_state_from_particles(
        dataset,
        main,
        optics,
        opts,
        rotations=np.zeros((3, 3, 3), dtype=np.float64),
    )

    np.testing.assert_array_equal(state.Iref, post_iref)
    np.testing.assert_array_equal(optics_groups, np.zeros(2, dtype=np.int64))
    assert calls == [
        {
            "pixel_size": 2.0,
            "ini_high_ang": state.ini_high,
            "particle_diameter_ang": 16.0,
            "width_mask_edge_px": float(opts.width_mask_edge_px),
            "do_init_blobs": True,
            "is_helical_segment": False,
        }
    ]


def test_native_expectation_step_rebuilds_sampling_per_iteration(monkeypatch):
    calls = []

    def fake_build_sampling_plan(opts, *, iteration):
        calls.append(iteration)
        return driver.NativeSamplingPlan(
            rotations=np.zeros((iteration, 3, 3), dtype=np.float32),
            translations=np.zeros((iteration + 1, 2), dtype=np.float32),
            random_perturbation=0.125,
        )

    def fake_run_dense(dataset, state, config, *, particle_ids, halfset_ids):
        assert config.rotations.shape == (3, 3, 3)
        assert config.translations.shape == (4, 2)
        np.testing.assert_array_equal(
            config.engine_kwargs["image_pre_shifts"],
            np.asarray([[1.0, -1.0], [0.0, 2.0]], dtype=np.float32),
        )
        assert particle_ids.tolist() == [0, 1]
        return SimpleNamespace(accumulators=["acc"], meta={})

    monkeypatch.setattr(driver, "_build_sampling_plan", fake_build_sampling_plan)
    monkeypatch.setattr(driver, "run_dense_initial_model_estep", fake_run_dense)
    dataset = SimpleNamespace(voxel_size=1.0, n_images=2)
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=3, n_directions=3)
    state.iter = 3

    expectation_step = driver._native_expectation_step(
        dataset,
        driver.NativeInitialModelOptions(fn_img="particles.star"),
        np.ones(33, dtype=np.float32),
        np.asarray([[1.0, -1.0], [0.0, 2.0]], dtype=np.float32),
    )
    accumulators, meta = expectation_step(state, np.asarray([0, 1]), np.asarray([0, 1], dtype=np.int8))

    assert accumulators == ["acc"]
    assert calls == [3]
    assert meta["random_perturbation"] == 0.125
    assert meta["n_rotations"] == 3
    assert meta["n_translations"] == 4


def test_native_expectation_step_updates_translation_offsets_between_iterations(monkeypatch):
    calls = []

    def fake_build_sampling_plan(opts, *, iteration):
        return driver.NativeSamplingPlan(
            rotations=np.zeros((1, 3, 3), dtype=np.float32),
            translations=np.asarray([[0.0, 0.0], [2.0, -1.0], [4.0, 0.0]], dtype=np.float32),
            random_perturbation=0.0,
        )

    def fake_run_dense(dataset, state, config, *, particle_ids, halfset_ids):
        calls.append(
            {
                "pre_shifts": np.asarray(config.engine_kwargs["image_pre_shifts"], dtype=np.float32).copy(),
                "prior": np.asarray(config.engine_kwargs["translation_log_prior"], dtype=np.float32).copy(),
            }
        )
        return SimpleNamespace(
            accumulators=[],
            meta={
                "selected_particle_ids": np.asarray([0, 1], dtype=np.int64),
                "pose_assignments": np.asarray([1, 2], dtype=np.int32),
                "class_assignments": np.asarray([1, 0], dtype=np.int32),
                "max_posterior_per_image": np.asarray([0.9, 0.8], dtype=np.float32),
            },
        )

    monkeypatch.setattr(driver, "_build_sampling_plan", fake_build_sampling_plan)
    monkeypatch.setattr(driver, "run_dense_initial_model_estep", fake_run_dense)
    dataset = SimpleNamespace(voxel_size=1.0, n_images=2)
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=2, n_directions=1)
    particle_state = driver.NativeParticleState(
        translation_offsets=np.asarray([[0.0, 0.0], [1.1, -1.0]], dtype=np.float32),
        class_assignments=np.zeros(2, dtype=np.int32),
        max_posterior=np.zeros(2, dtype=np.float32),
    )
    expectation_step = driver._native_expectation_step(
        dataset,
        driver.NativeInitialModelOptions(fn_img="particles.star", translation_sigma_angstrom=2.0),
        np.ones(33, dtype=np.float32),
        particle_state,
    )

    expectation_step(state, np.asarray([0, 1]), np.asarray([0, 1], dtype=np.int8))
    expectation_step(state, np.asarray([0, 1]), np.asarray([0, 1], dtype=np.int8))

    np.testing.assert_array_equal(calls[0]["pre_shifts"], np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32))
    np.testing.assert_array_equal(calls[1]["pre_shifts"], np.asarray([[2.0, -1.0], [5.0, -1.0]], dtype=np.float32))
    assert calls[0]["prior"].shape == (2, 3)
    assert calls[1]["prior"].shape == (2, 3)
    assert calls[0]["prior"][0, 0] == pytest.approx(0.0)
    assert calls[1]["prior"][0, 0] < calls[0]["prior"][0, 0]
    np.testing.assert_array_equal(
        particle_state.translation_offsets,
        np.asarray([[4.0, -2.0], [9.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(particle_state.class_assignments, [1, 0])
    np.testing.assert_allclose(particle_state.max_posterior, [0.9, 0.8])
    np.testing.assert_array_equal(particle_state.pose_assignments, [1, 2])


def test_update_particle_state_preserves_best_pose_metadata():
    particle_state = driver.NativeParticleState(
        translation_offsets=np.zeros((3, 2), dtype=np.float32),
        class_assignments=np.zeros(3, dtype=np.int32),
        max_posterior=np.zeros(3, dtype=np.float32),
        pose_assignments=np.full(3, -1, dtype=np.int32),
    )
    rotations = np.stack(
        [
            np.eye(3, dtype=np.float32),
            np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        ],
        axis=0,
    )

    driver._update_particle_state_from_estep_meta(
        particle_state,
        {
            "selected_particle_ids": np.asarray([2, 0], dtype=np.int64),
            "pose_assignments": np.asarray([1, 0], dtype=np.int32),
            "best_pose_rotations": rotations,
            "best_pose_translations": np.asarray([[3.0, -1.0], [0.0, 2.0]], dtype=np.float32),
            "best_pose_rotation_ids": np.asarray([11, 7], dtype=np.int32),
            "healpix_order": 1,
            "oversampling": 1,
        },
        np.asarray([[0.0, 2.0], [3.0, -1.0]], dtype=np.float32),
    )

    np.testing.assert_array_equal(particle_state.pose_assignments, [0, -1, 1])
    np.testing.assert_allclose(particle_state.best_pose_rotations[[2, 0]], rotations)
    np.testing.assert_allclose(
        particle_state.best_pose_translations,
        np.asarray([[0.0, 2.0], [0.0, 0.0], [3.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(particle_state.best_pose_rotation_ids, [7, -1, 11])
    np.testing.assert_array_equal(particle_state.best_pose_rotation_orders, [2, -1, 2])
    np.testing.assert_array_equal(particle_state.visited, [True, False, True])


def test_best_eulers_from_particle_state_prefers_stored_rotation_matrices():
    grid_eulers = driver.sampling.get_relion_rotation_grid_eulers(1, rotation_index_order="relion")
    grid_rotations = driver.sampling.get_relion_rotation_grid(1, rotation_index_order="relion")
    perturbed_euler = np.asarray([[33.0, 44.0, 55.0]], dtype=np.float64)
    perturbed_rotation = driver.sampling._relion_euler_angles_to_matrix(perturbed_euler)[0].astype(np.float32)
    particle_state = driver.NativeParticleState(
        translation_offsets=np.zeros((2, 2), dtype=np.float32),
        class_assignments=np.zeros(2, dtype=np.int32),
        max_posterior=np.ones(2, dtype=np.float32),
        best_pose_rotations=np.stack([perturbed_rotation, grid_rotations[7]], axis=0),
        best_pose_rotation_ids=np.asarray([5, 7], dtype=np.int32),
    )

    eulers = driver._best_eulers_from_particle_state(
        particle_state,
        np.asarray([0, 1], dtype=np.int64),
        rotation_grid_order=1,
    )

    assert eulers is not None
    assert not np.allclose(eulers[0], grid_eulers[5])
    np.testing.assert_allclose(
        driver.sampling._relion_euler_angles_to_matrix(eulers),
        np.stack([perturbed_rotation, grid_rotations[7]], axis=0),
        atol=1e-5,
    )


def test_native_expectation_step_uses_autosampling_state_at_iteration_ten(monkeypatch):
    build_calls = []

    def fake_build_sampling_plan(opts, *, iteration, sampling_state=None):
        assert sampling_state is not None
        build_calls.append((iteration, sampling_state.healpix_order, sampling_state.offset_range_angstrom))
        return driver.NativeSamplingPlan(
            rotations=np.zeros((2, 3, 3), dtype=np.float32),
            translations=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            random_perturbation=0.0,
            healpix_order=sampling_state.healpix_order,
            oversampling=sampling_state.adaptive_oversampling,
            offset_range_px=sampling_state.offset_range_px,
            offset_step_px=sampling_state.offset_step_px,
            offset_range_angstrom=sampling_state.offset_range_angstrom,
            offset_step_angstrom=sampling_state.offset_step_angstrom,
        )

    def fake_run_dense(dataset, state, config, *, particle_ids, halfset_ids):
        assert config.engine_kwargs["healpix_order"] == 2
        return SimpleNamespace(
            accumulators=[],
            meta={
                "selected_particle_ids": np.asarray([0], dtype=np.int64),
                "pose_assignments": np.asarray([1], dtype=np.int32),
                "class_assignments": np.asarray([0], dtype=np.int32),
                "max_posterior_per_image": np.asarray([0.75], dtype=np.float32),
            },
        )

    monkeypatch.setattr(driver, "_build_sampling_plan", fake_build_sampling_plan)
    monkeypatch.setattr(driver, "run_dense_initial_model_estep", fake_run_dense)

    opts = driver.NativeInitialModelOptions(fn_img="particles.star", nr_iter=200)
    sampling_state = driver._initial_sampling_state(opts, pixel_size=2.125)
    particle_state = driver.NativeParticleState(
        translation_offsets=np.zeros((1, 2), dtype=np.float32),
        class_assignments=np.zeros(1, dtype=np.int32),
        max_posterior=np.zeros(1, dtype=np.float32),
        pose_assignments=np.full(1, -1, dtype=np.int32),
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=2.125, K=1, nr_iter=200, n_directions=1)
    state.iter = 10
    sampling_state.last_current_resolution = float(state.current_resolution)

    expectation_step = driver._native_expectation_step(
        SimpleNamespace(voxel_size=2.125, n_images=1),
        opts,
        np.ones(5, dtype=np.float32),
        particle_state,
        sampling_state,
    )
    _accumulators, meta = expectation_step(state, np.asarray([0]), np.asarray([0], dtype=np.int8))

    assert build_calls == [(10, 2, pytest.approx(8.2875))]
    assert meta["sampling_updated"] is True
    assert meta["healpix_order"] == 2
    assert meta["offset_range_angstrom"] == pytest.approx(8.2875)
    assert meta["offset_step_angstrom"] == pytest.approx(3.0)
    assert meta["current_changes_optimal_offsets_angstrom"] == pytest.approx(2.125 / np.sqrt(2.0))


def test_native_expectation_step_estimates_sampling_accuracy_before_update(monkeypatch):
    build_calls = []
    estimate_calls = []

    def fake_estimate_sampling_accuracy(
        sampling_state,
        state,
        particle_state,
        optics_state,
        *,
        particle_order,
        random_seed,
        padding_factor,
    ):
        estimate_calls.append(
            {
                "healpix_order": sampling_state.healpix_order,
                "offset_range_angstrom": sampling_state.offset_range_angstrom,
                "particle_order": np.asarray(particle_order, dtype=np.int64).copy(),
                "random_seed": random_seed,
                "padding_factor": padding_factor,
            }
        )
        sampling_state.acc_rot = 3.666
        sampling_state.acc_trans_angstrom = 2.125
        return {"estimated_acc_rot": 3.666, "estimated_acc_trans_angstrom": 2.125}

    def fake_build_sampling_plan(opts, *, iteration, sampling_state=None):
        assert sampling_state is not None
        build_calls.append((iteration, sampling_state.healpix_order, sampling_state.offset_range_angstrom))
        return driver.NativeSamplingPlan(
            rotations=np.zeros((2, 3, 3), dtype=np.float32),
            translations=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            random_perturbation=0.0,
            healpix_order=sampling_state.healpix_order,
            oversampling=sampling_state.adaptive_oversampling,
            offset_range_px=sampling_state.offset_range_px,
            offset_step_px=sampling_state.offset_step_px,
            offset_range_angstrom=sampling_state.offset_range_angstrom,
            offset_step_angstrom=sampling_state.offset_step_angstrom,
        )

    def fake_run_dense(dataset, state, config, *, particle_ids, halfset_ids):
        assert config.engine_kwargs["healpix_order"] == 2
        assert config.translations.shape == (2, 2)
        return SimpleNamespace(
            accumulators=[],
            meta={
                "selected_particle_ids": np.asarray([0, 1], dtype=np.int64),
                "pose_assignments": np.asarray([0, 1], dtype=np.int32),
                "class_assignments": np.asarray([0, 0], dtype=np.int32),
                "max_posterior_per_image": np.asarray([0.8, 0.7], dtype=np.float32),
            },
        )

    monkeypatch.setattr(driver, "_estimate_native_sampling_accuracy", fake_estimate_sampling_accuracy)
    monkeypatch.setattr(driver, "_build_sampling_plan", fake_build_sampling_plan)
    monkeypatch.setattr(driver, "run_dense_initial_model_estep", fake_run_dense)

    opts = driver.NativeInitialModelOptions(fn_img="particles.star", nr_iter=200, random_seed=17, padding_factor=2)
    sampling_state = driver._initial_sampling_state(opts, pixel_size=2.125)
    sampling_state.current_changes_optimal_offsets_angstrom = 10.366644 / 5.0
    particle_state = driver.NativeParticleState(
        translation_offsets=np.zeros((2, 2), dtype=np.float32),
        class_assignments=np.zeros(2, dtype=np.int32),
        max_posterior=np.zeros(2, dtype=np.float32),
        pose_assignments=np.full(2, -1, dtype=np.int32),
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=2.125, K=1, nr_iter=200, n_directions=1)
    state.iter = 10
    sampling_state.last_current_resolution = float(state.current_resolution)

    optics_state = driver.NativeOpticsState(
        voltage=300.0,
        Cs=2.7,
        Q0=0.07,
        pixel_size=2.125,
        defU=np.full(2, 10000.0, dtype=np.float64),
        defV=np.full(2, 10000.0, dtype=np.float64),
        defAngle=np.zeros(2, dtype=np.float64),
        phase_shift=np.zeros(2, dtype=np.float64),
    )
    expectation_step = driver._native_expectation_step(
        SimpleNamespace(voxel_size=2.125, n_images=2),
        opts,
        np.ones(5, dtype=np.float32),
        particle_state,
        sampling_state,
        optics_state,
    )
    _accumulators, meta = expectation_step(state, np.asarray([1, 0]), np.asarray([0, 1], dtype=np.int8))

    assert len(estimate_calls) == 1
    assert estimate_calls[0]["healpix_order"] == 1
    assert estimate_calls[0]["offset_range_angstrom"] == pytest.approx(12.75)
    np.testing.assert_array_equal(estimate_calls[0]["particle_order"], np.asarray([1, 0], dtype=np.int64))
    assert estimate_calls[0]["random_seed"] == 17
    assert estimate_calls[0]["padding_factor"] == 2
    assert build_calls == [(10, 2, pytest.approx(10.366644))]
    assert meta["sampling_accuracy_estimated"] is True
    assert meta["estimated_acc_rot"] == pytest.approx(3.666)
    assert meta["estimated_acc_trans_angstrom"] == pytest.approx(2.125)
    assert meta["sampling_acc_rot"] == pytest.approx(3.666)
    assert meta["sampling_acc_trans_angstrom"] == pytest.approx(2.125)
    assert meta["offset_range_angstrom"] == pytest.approx(10.366644)
    assert meta["offset_step_angstrom"] == pytest.approx(3.0)


def test_native_expectation_step_records_sampling_changes_each_gradient_iteration(monkeypatch):
    build_calls = []

    def fake_build_sampling_plan(opts, *, iteration, sampling_state=None):
        build_calls.append(iteration)
        return driver.NativeSamplingPlan(
            rotations=np.zeros((2, 3, 3), dtype=np.float32),
            translations=np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            random_perturbation=0.0,
            healpix_order=1 if sampling_state is None else sampling_state.healpix_order,
            oversampling=0,
            offset_range_px=6.0,
            offset_step_px=2.0,
            offset_range_angstrom=12.0,
            offset_step_angstrom=4.0,
        )

    def fake_run_dense(dataset, state, config, *, particle_ids, halfset_ids):
        return SimpleNamespace(
            accumulators=[],
            meta={
                "selected_particle_ids": np.asarray([0, 1], dtype=np.int64),
                "pose_assignments": np.asarray([1, 0], dtype=np.int32),
                "class_assignments": np.asarray([0, 0], dtype=np.int32),
                "max_posterior_per_image": np.asarray([0.8, 0.7], dtype=np.float32),
            },
        )

    monkeypatch.setattr(driver, "_build_sampling_plan", fake_build_sampling_plan)
    monkeypatch.setattr(driver, "run_dense_initial_model_estep", fake_run_dense)

    opts = driver.NativeInitialModelOptions(fn_img="particles.star", nr_iter=200, oversampling=0)
    sampling_state = driver._initial_sampling_state(opts, pixel_size=2.0)
    particle_state = driver.NativeParticleState(
        translation_offsets=np.zeros((2, 2), dtype=np.float32),
        class_assignments=np.zeros(2, dtype=np.int32),
        max_posterior=np.zeros(2, dtype=np.float32),
        pose_assignments=np.full(2, -1, dtype=np.int32),
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=2.0, K=1, nr_iter=200, n_directions=1)
    state.iter = 9
    sampling_state.last_current_resolution = float(state.current_resolution)

    expectation_step = driver._native_expectation_step(
        SimpleNamespace(voxel_size=2.0, n_images=2),
        opts,
        np.ones(5, dtype=np.float32),
        particle_state,
        sampling_state,
    )
    _accumulators, meta = expectation_step(state, np.asarray([0, 1]), np.asarray([0, 1], dtype=np.int8))

    assert build_calls == [9]
    assert meta["sampling_updated"] is False
    assert meta["current_changes_optimal_offsets_angstrom"] == pytest.approx(2.0)
    assert sampling_state.current_changes_optimal_offsets_angstrom == pytest.approx(2.0)


def test_native_expectation_step_expands_class_rotation_prior_for_dense_fallback(monkeypatch):
    captured = {}

    monkeypatch.setenv("RECOVAR_DISABLE_SPARSE_PASS2", "1")

    def fake_build_sampling_plan(opts, *, iteration, sampling_state=None):
        return driver.NativeSamplingPlan(
            rotations=np.zeros((4, 3, 3), dtype=np.float32),
            translations=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            random_perturbation=0.0,
            healpix_order=1,
            oversampling=1,
            offset_range_px=1.0,
            offset_step_px=1.0,
            offset_range_angstrom=1.0,
            offset_step_angstrom=1.0,
        )

    def fake_class_direction_rotation_log_prior(state, healpix_order):
        assert healpix_order == 1
        return np.asarray([[0.25, 0.75]], dtype=np.float32)

    def fake_expand(prior, sampling_plan):
        assert sampling_plan.oversampling == 1
        np.testing.assert_allclose(prior, np.asarray([[0.25, 0.75]], dtype=np.float32))
        return np.asarray([[0.25, 0.75, 0.25, 0.75]], dtype=np.float32)

    def fake_run_dense(dataset, state, config, *, particle_ids, halfset_ids):
        captured["sparse_pass2"] = bool(config.engine_kwargs["sparse_pass2"])
        captured["max_significants"] = int(config.engine_kwargs["max_significants"])
        captured["class_rotation_log_prior"] = np.asarray(config.engine_kwargs["class_rotation_log_prior"])
        return SimpleNamespace(
            accumulators=[],
            meta={
                "selected_particle_ids": np.asarray([0, 1], dtype=np.int64),
                "pose_assignments": np.asarray([1, 0], dtype=np.int32),
                "class_assignments": np.asarray([0, 0], dtype=np.int32),
                "max_posterior_per_image": np.asarray([0.8, 0.7], dtype=np.float32),
            },
        )

    monkeypatch.setattr(driver, "_build_sampling_plan", fake_build_sampling_plan)
    monkeypatch.setattr(driver, "_class_direction_rotation_log_prior", fake_class_direction_rotation_log_prior)
    monkeypatch.setattr(driver, "_expand_class_rotation_log_prior_for_dense_fine_grid", fake_expand)
    monkeypatch.setattr(driver, "run_dense_initial_model_estep", fake_run_dense)

    opts = driver.NativeInitialModelOptions(fn_img="particles.star", oversampling=1)
    particle_state = driver.NativeParticleState(
        translation_offsets=np.zeros((2, 2), dtype=np.float32),
        class_assignments=np.zeros(2, dtype=np.int32),
        max_posterior=np.zeros(2, dtype=np.float32),
        pose_assignments=np.full(2, -1, dtype=np.int32),
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=2.0, K=1, nr_iter=1, n_directions=1)
    expectation_step = driver._native_expectation_step(
        SimpleNamespace(voxel_size=2.0, n_images=2),
        opts,
        np.ones(5, dtype=np.float32),
        particle_state,
    )

    _accumulators, _meta = expectation_step(state, np.asarray([0, 1]), np.asarray([0, 1], dtype=np.int8))

    assert captured["sparse_pass2"] is False
    assert captured["max_significants"] == 100
    np.testing.assert_allclose(
        captured["class_rotation_log_prior"],
        np.asarray([[0.25, 0.75, 0.25, 0.75]], dtype=np.float32),
    )


def test_expand_class_rotation_prior_for_dense_fine_grid_uses_parent_map(monkeypatch):
    prior = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    parent_map = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    def fake_oversampled(parent_rotation_indices, parent_nside_level, oversampling_order, *, random_perturbation):
        np.testing.assert_array_equal(parent_rotation_indices, np.arange(3, dtype=np.int64))
        assert parent_nside_level == 2
        assert oversampling_order == 1
        assert random_perturbation == pytest.approx(0.125)
        return np.zeros((6, 3, 3), dtype=np.float32), parent_map

    monkeypatch.setattr(
        driver.sampling,
        "get_oversampled_relion_hidden_rotation_grid_from_samples",
        fake_oversampled,
    )

    plan = driver.NativeSamplingPlan(
        rotations=np.zeros((6, 3, 3), dtype=np.float32),
        translations=np.zeros((1, 2), dtype=np.float32),
        random_perturbation=0.125,
        healpix_order=2,
        oversampling=1,
    )

    expanded = driver._expand_class_rotation_log_prior_for_dense_fine_grid(prior, plan)

    np.testing.assert_allclose(expanded, prior[:, parent_map])


def test_dense_estep_config_splits_fine_and_coarse_translation_priors():
    dataset = SimpleNamespace(voxel_size=2.0, n_images=1)
    opts = driver.NativeInitialModelOptions(
        fn_img="particles.star",
        oversampling=1,
        translation_sigma_angstrom=4.0,
    )
    plan = driver.NativeSamplingPlan(
        rotations=np.zeros((1, 3, 3), dtype=np.float32),
        translations=np.asarray([[0.5, 0.0], [1.5, 0.0]], dtype=np.float32),
        random_perturbation=0.0,
        coarse_translations=np.asarray([[99.0, 0.0]], dtype=np.float32),
        coarse_prior_translations=np.asarray([[1.0, 0.0]], dtype=np.float32),
        translation_parent=np.asarray([0, 0], dtype=np.int64),
    )

    config = driver._dense_estep_config(
        dataset,
        opts,
        np.ones(5, dtype=np.float32),
        plan,
        np.zeros((1, 2), dtype=np.float32),
    )

    fine_prior = np.asarray(config.engine_kwargs["translation_log_prior"], dtype=np.float32)
    coarse_prior = np.asarray(config.engine_kwargs["coarse_translation_log_prior"], dtype=np.float32)
    np.testing.assert_allclose(fine_prior, np.asarray([[-0.125, -1.125]], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(coarse_prior, np.asarray([[-0.5]], dtype=np.float32), rtol=1e-6)


def test_driver_output_mrc_path_matches_relion_snapshot():
    assert driver._initial_model_mrc_from_prefix("ab_initio/run") == "ab_initio/initial_model.mrc"


def test_model_star_uses_relion_model_blocks(tmp_path):
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=2, nr_iter=1, n_directions=12)
    state.pdf_class = np.asarray([0.25, 0.75], dtype=np.float64)
    state.iter = 3
    state.current_size = 6
    state.current_resolution = 0.375
    state.tau2_fudge_factor = 3.5
    state.ave_Pmax = 0.625
    state.sigma2_offset = 49.0
    state.tau2_class[:] = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    state.data_vs_prior_class[:] = np.asarray([[10.0, 9.0, 8.0, 7.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
    state.sigma2_class[:] = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]])
    state.fourier_coverage_class[:] = np.asarray(
        [[0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]]
    )
    out = tmp_path / "run_it001_model.star"

    driver._write_model_star(str(out), state, ("run_it001_class001.mrc", "run_it001_class002.mrc"))

    text = out.read_text()
    assert "data_model_general" in text
    assert "data_model_classes" in text
    assert "data_model_class_1" in text
    assert "data_model_class_2" in text
    assert "data_model_pdf_orient_class_1" in text
    assert "data_model_pdf_orient_class_2" in text
    assert "data_model_optics_group_1" in text
    assert "_rlnCurrentImageSize 6" in text
    assert "_rlnCurrentResolution 2.66666666667" in text
    assert "_rlnCurrentIteration 3" in text
    assert "_rlnTau2FudgeFactor 3.5" in text
    assert "_rlnAveragePmax 0.625" in text
    assert "_rlnSigmaOffsetsAngst 7" in text
    assert "_rlnSsnrMap" in text
    assert "_rlnReferenceTau2" in text
    assert "_rlnReferenceSigma2" in text
    assert "_rlnFourierCompleteness" in text
    assert "_rlnReferenceImage" in text
    assert "run_it001_class001.mrc 0.25 2.66666666667" in text
    assert "run_it001_class002.mrc 0.75 2.66666666667" in text
    assert "1 0.125 8 9 0 0.8 0.2 2" in text
    assert "1 0.125 8 2 0 0.2 0.4 4" in text
    assert "_rlnOrientationDistribution" in text


def test_data_star_preserves_optics_and_updates_particle_metadata(tmp_path):
    main = pd.DataFrame(
        {
            "_rlnImageName": ["2@stack.mrcs", "1@stack.mrcs"],
            "_rlnMicrographName": ["2", "1"],
            "_rlnOpticsGroup": ["1", "1"],
            "_rlnOriginXAngst": ["0.0", "0.0"],
            "_rlnOriginYAngst": ["0.0", "0.0"],
            "_rlnOriginX": ["0.0", "0.0"],
            "_rlnOriginY": ["0.0", "0.0"],
        }
    )
    optics = pd.DataFrame({"_rlnOpticsGroup": ["1"], "_rlnImageSize": ["8"]})
    particle_state = driver.NativeParticleState(
        translation_offsets=np.asarray([[2.0, -1.0], [0.5, 1.25]], dtype=np.float32),
        class_assignments=np.asarray([1, 0], dtype=np.int32),
        max_posterior=np.asarray([0.875, 0.25], dtype=np.float32),
    )
    out = tmp_path / "run_it001_data.star"

    driver._write_data_star(
        str(out),
        main,
        optics,
        SimpleNamespace(voxel_size=1.5, n_images=2),
        particle_state,
    )

    data, data_optics = read_star(str(out))
    assert data_optics is not None
    assert data_optics["_rlnImageSize"].tolist() == ["8"]
    assert data["_rlnImageName"].tolist() == ["1@stack.mrcs", "2@stack.mrcs"]
    np.testing.assert_allclose(data["_rlnOriginXAngst"].astype(float).to_numpy(), [0.75, 3.0])
    np.testing.assert_allclose(data["_rlnOriginYAngst"].astype(float).to_numpy(), [1.875, -1.5])
    np.testing.assert_allclose(data["_rlnOriginX"].astype(float).to_numpy(), [0.5, 2.0])
    np.testing.assert_allclose(data["_rlnOriginY"].astype(float).to_numpy(), [1.25, -1.0])
    np.testing.assert_array_equal(data["_rlnClassNumber"].astype(int).to_numpy(), [1, 2])
    np.testing.assert_allclose(data["_rlnMaxValueProbDistribution"].astype(float).to_numpy(), [0.25, 0.875])


def test_data_star_zeros_unvisited_rows_and_writes_best_pose_eulers(tmp_path):
    main = pd.DataFrame(
        {
            "_rlnImageName": ["3@stack.mrcs", "1@stack.mrcs", "2@stack.mrcs"],
            "_rlnMicrographName": ["3", "1", "2"],
            "_rlnOpticsGroup": ["1", "1", "1"],
            "_rlnAngleRot": ["10.0", "20.0", "30.0"],
            "_rlnAngleTilt": ["11.0", "21.0", "31.0"],
            "_rlnAnglePsi": ["12.0", "22.0", "32.0"],
            "_rlnOriginXAngst": ["0.0", "0.0", "0.0"],
            "_rlnOriginYAngst": ["0.0", "0.0", "0.0"],
            "_rlnClassNumber": ["1", "0", "0"],
            "_rlnMaxValueProbDistribution": ["0.5", "0.0", "0.0"],
        }
    )
    particle_state = driver.NativeParticleState(
        translation_offsets=np.zeros((3, 2), dtype=np.float32),
        class_assignments=np.asarray([0, 0, 0], dtype=np.int32),
        max_posterior=np.asarray([0.75, 0.0, 0.625], dtype=np.float32),
        best_pose_rotation_ids=np.asarray([5, -1, 9], dtype=np.int32),
        best_pose_rotation_orders=np.asarray([1, -1, 1], dtype=np.int32),
        visited=np.asarray([True, False, True]),
    )
    out = tmp_path / "run_it010_data.star"

    driver._write_data_star(
        str(out),
        main,
        None,
        SimpleNamespace(voxel_size=1.0, n_images=3),
        particle_state,
    )

    data, _ = read_star(str(out))
    expected_eulers = driver.sampling.get_relion_rotation_grid_eulers(1, rotation_index_order="relion")
    assert data["_rlnImageName"].tolist() == ["1@stack.mrcs", "2@stack.mrcs", "3@stack.mrcs"]
    np.testing.assert_array_equal(data["_rlnClassNumber"].astype(int).to_numpy(), [0, 1, 1])
    np.testing.assert_allclose(data["_rlnMaxValueProbDistribution"].astype(float).to_numpy(), [0.0, 0.625, 0.75])
    np.testing.assert_allclose(data["_rlnAngleRot"].astype(float).to_numpy()[[1, 2]], expected_eulers[[9, 5], 0])
    np.testing.assert_allclose(data["_rlnAngleTilt"].astype(float).to_numpy()[[1, 2]], expected_eulers[[9, 5], 1])
    np.testing.assert_allclose(data["_rlnAnglePsi"].astype(float).to_numpy()[[1, 2]], expected_eulers[[9, 5], 2])
    np.testing.assert_allclose(data["_rlnAngleRot"].astype(float).to_numpy()[0], 20.0)
    np.testing.assert_allclose(data["_rlnAngleTilt"].astype(float).to_numpy()[0], 21.0)
    np.testing.assert_allclose(data["_rlnAnglePsi"].astype(float).to_numpy()[0], 22.0)


def test_cli_non_dry_run_calls_native_driver(monkeypatch, capsys):
    run_ab_initio = _load_run_ab_initio()
    calls = {}

    def fake_run_native(opts):
        calls["opts"] = opts
        return SimpleNamespace(final_mrc="out/initial_model.mrc", final_model_star="out/run_it003_model.star")

    monkeypatch.setattr(driver, "run_native_initial_model", fake_run_native)

    rc = run_ab_initio.main(
        [
            "--i",
            "particles.star",
            "--o",
            "out/run",
            "--nr_iter",
            "3",
            "--K",
            "2",
            "--particle_diameter",
            "250",
            "--random_seed",
            "17",
            "--healpix_order",
            "2",
            "--oversampling",
            "0",
            "--offset_range",
            "4.5",
            "--offset_step",
            "1.5",
            "--random_perturbation",
            "0.25",
            "--translation_sigma_angstrom",
            "6.5",
            "--no_iter_artifacts",
        ]
    )

    assert rc == 0
    opts = calls["opts"]
    assert opts.fn_img == "particles.star"
    assert opts.outputname == "out/run"
    assert opts.nr_iter == 3
    assert opts.nr_classes == 2
    assert opts.particle_diameter == 250.0
    assert opts.random_seed == 17
    assert opts.healpix_order == 2
    assert opts.oversampling == 0
    assert opts.offset_range_px == 4.5
    assert opts.offset_step_px == 1.5
    assert opts.random_perturbation == 0.25
    assert opts.translation_sigma_angstrom == 6.5
    assert opts.write_iter_artifacts is False
    assert "recovar InitialModel complete: out/initial_model.mrc" in capsys.readouterr().out
