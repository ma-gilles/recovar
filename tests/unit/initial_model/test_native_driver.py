"""Native InitialModel driver tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import recovar.em.initial_model.driver as driver
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


def test_translation_log_prior_uses_angstrom_distance():
    translations = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, -1.0]], dtype=np.float32)

    prior = driver._translation_log_prior(translations, voxel_size=3.0, sigma_angstrom=6.0)

    np.testing.assert_allclose(prior, np.asarray([0.0, -0.5, -0.125], dtype=np.float32), rtol=1e-6)


def test_image_pre_shifts_from_star_converts_angstrom_origins_to_rounded_pixels():
    main = pd.DataFrame(
        {
            "_rlnOriginXAngst": ["4.2", "-3.9", "0.0"],
            "_rlnOriginYAngst": ["-8.1", "2.0", "0.4"],
        }
    )

    shifts = driver._image_pre_shifts_from_star(main, SimpleNamespace(voxel_size=2.0))

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


def test_driver_output_mrc_path_matches_relion_snapshot():
    assert driver._initial_model_mrc_from_prefix("ab_initio/run") == "ab_initio/initial_model.mrc"


def test_model_star_uses_relion_model_blocks(tmp_path):
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=2, nr_iter=1, n_directions=12)
    state.pdf_class = np.asarray([0.25, 0.75], dtype=np.float64)
    out = tmp_path / "run_it001_model.star"

    driver._write_model_star(str(out), state, ("run_it001_class001.mrc", "run_it001_class002.mrc"))

    text = out.read_text()
    assert "data_model_classes" in text
    assert "data_model_optics_group_1" in text
    assert "_rlnReferenceImage" in text
    assert "run_it001_class001.mrc 0.25 0" in text
    assert "run_it001_class002.mrc 0.75 0" in text


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
            "--oversampling",
            "0",
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
    assert opts.oversampling == 0
    assert opts.random_perturbation == 0.25
    assert opts.translation_sigma_angstrom == 6.5
    assert opts.write_iter_artifacts is False
    assert "recovar InitialModel complete: out/initial_model.mrc" in capsys.readouterr().out
