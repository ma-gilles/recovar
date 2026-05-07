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


def test_micrograph_sort_order_is_stable():
    main = pd.DataFrame(
        {
            "_rlnMicrographName": ["b", "a", "b", "a"],
            "_rlnImageName": ["1@s.mrcs", "2@s.mrcs", "3@s.mrcs", "4@s.mrcs"],
        }
    )

    assert driver._micrograph_sort_order(main).tolist() == [1, 3, 0, 2]


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
    out = tmp_path / "run_it001_model.star"

    driver._write_model_star(str(out), state, ("run_it001_class001.mrc", "run_it001_class002.mrc"))

    text = out.read_text()
    assert "data_model_general" in text
    assert "data_model_classes" in text
    assert "data_model_optics_group_1" in text
    assert "_rlnCurrentImageSize 6" in text
    assert "_rlnCurrentResolution 0.375" in text
    assert "_rlnCurrentIteration 3" in text
    assert "_rlnTau2FudgeFactor 3.5" in text
    assert "_rlnAveragePmax 0.625" in text
    assert "_rlnReferenceImage" in text
    assert "run_it001_class001.mrc 0.25 0" in text
    assert "run_it001_class002.mrc 0.75 0" in text


def test_data_star_preserves_optics_and_updates_particle_metadata(tmp_path):
    main = pd.DataFrame(
        {
            "_rlnImageName": ["1@stack.mrcs", "2@stack.mrcs"],
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
    np.testing.assert_allclose(data["_rlnOriginXAngst"].astype(float).to_numpy(), [3.0, 0.75])
    np.testing.assert_allclose(data["_rlnOriginYAngst"].astype(float).to_numpy(), [-1.5, 1.875])
    np.testing.assert_allclose(data["_rlnOriginX"].astype(float).to_numpy(), [2.0, 0.5])
    np.testing.assert_allclose(data["_rlnOriginY"].astype(float).to_numpy(), [-1.0, 1.25])
    np.testing.assert_array_equal(data["_rlnClassNumber"].astype(int).to_numpy(), [2, 1])
    np.testing.assert_allclose(data["_rlnMaxValueProbDistribution"].astype(float).to_numpy(), [0.875, 0.25])


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
