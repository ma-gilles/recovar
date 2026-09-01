"""Native InitialModel driver tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import recovar.em.initial_model.driver as driver
from recovar.data_io.starfile import read_star
from recovar.em.dense_single_volume.batch_planning import maybe_cache_raw_image_loaders
from recovar.em.initial_model import initialise_denovo_state
from recovar.em.initial_model.iteration_loop import select_subset_for_iter

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


def test_noise_variance_preserves_relion_rfloat_shell_values():
    sigma2 = np.asarray([[1.00000006e-5, 2.00000012e-5, 3.00000018e-5]], dtype=np.float64)

    noise = driver._noise_variance_from_sigma2(sigma2, 4)

    assert noise.dtype == np.float64
    assert np.any(noise != noise.astype(np.float32).astype(np.float64))


@pytest.mark.parametrize(
    ("requested", "grid_size", "gpu_memory_gb", "expected"),
    [
        (500, 128, 40.0, 500),
        (500, 256, 40.0, 32),
        (500, 256, 80.0, 64),
        (500, 384, 40.0, 14),
        (25, 384, 80.0, 25),
        (8, 256, 40.0, 8),
    ],
)
def test_effective_initial_model_image_batch_size(requested, grid_size, gpu_memory_gb, expected):
    assert (
        driver._effective_initial_model_image_batch_size(
            requested,
            grid_size=grid_size,
            gpu_memory_gb=gpu_memory_gb,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("iteration", "nr_iter", "grad_write_iter", "expected"),
    [(1, 25, 10, False), (10, 25, 10, True), (20, 25, 10, True), (25, 25, 10, True)],
)
def test_iteration_artifact_cadence_matches_relion(iteration, nr_iter, grad_write_iter, expected):
    assert driver._should_write_iteration_artifacts(iteration, nr_iter, grad_write_iter) is expected


def test_iteration_artifact_cadence_rejects_nonpositive_interval():
    with pytest.raises(ValueError, match="grad_write_iter must be >= 1"):
        driver._should_write_iteration_artifacts(1, 10, 0)


def _write_test_mrc(path: Path, values: np.ndarray) -> None:
    import mrcfile

    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(values, dtype=np.float32))


def _write_native_vdam_checkpoint(tmp_path: Path) -> tuple[Path, Path, dict[str, np.ndarray]]:
    ori_size = 4
    n_shells = 3
    reference = np.arange(ori_size**3, dtype=np.float32).reshape((ori_size,) * 3)
    moment_values = {}
    for index, scale in ((1, 1.0), (2, 2.0)):
        complex_values = (
            np.arange(ori_size * ori_size * n_shells, dtype=np.float32).reshape(
                ori_size,
                ori_size,
                n_shells,
            )
            * scale
            + 1j * scale
        ).astype(np.complex64)
        moment_values[f"moment{index}"] = complex_values
        _write_test_mrc(
            tmp_path / f"run_it180_1moment{index:03d}.mrc",
            complex_values.view(np.float32).reshape(ori_size, ori_size, 2 * n_shells),
        )
    second_moment = (moment_values["moment1"] * 3.0 + 2.0j).astype(np.complex64)
    moment_values["second_moment"] = second_moment
    _write_test_mrc(
        tmp_path / "run_it180_2moment001.mrc",
        second_moment.view(np.float32).reshape(ori_size, ori_size, 2 * n_shells),
    )
    _write_test_mrc(tmp_path / "run_it180_class001.mrc", reference)

    model = tmp_path / "run_it180_model.star"
    model.write_text(
        """data_model_general
_rlnOriginalImageSize 4
_rlnCurrentResolution 8
_rlnCurrentImageSize 4
_rlnPaddingFactor 1
_rlnPixelSize 2
_rlnNrClasses 1
_rlnTau2FudgeFactor 4
_rlnSigmaOffsetsAngst 5
_rlnAveragePmax 0.75
_rlnOrientationalPriorMode 1
_rlnSigmaPriorRotAngle 0
_rlnSigmaPriorTiltAngle 0
_rlnSigmaPriorPsiAngle 0

data_model_classes

loop_
_rlnReferenceImage #1
_rlnGradMoment1 #2
_rlnGradMoment2 #3
_rlnClassDistribution #4
_rlnAccuracyRotations #5
_rlnAccuracyTranslationsAngst #6
_rlnEstimatedResolution #7
_rlnOverallFourierCompleteness #8
run_it180_class001.mrc run_it180_1moment001.mrc run_it180_2moment001.mrc 1 0.1 0.4 8 1

data_model_class_1

loop_
_rlnSpectralIndex #1
_rlnResolution #2
_rlnAngstromResolution #3
_rlnSsnrMap #4
_rlnGoldStandardFsc #5
_rlnFourierCompleteness #6
_rlnReferenceSigma2 #7
_rlnReferenceTau2 #8
_rlnSpectralOrientabilityContribution #9
0 0 999 10 0.1 0.9 1 4 0
1 0.125 8 9 0.2 0.8 2 5 0
2 0.25 4 8 0.3 0.7 3 6 0

data_model_optics_group_1

loop_
_rlnSpectralIndex #1
_rlnResolution #2
_rlnSigma2Noise #3
0 0 0.01
1 0.125 0.02
2 0.25 0.03

data_model_pdf_orient_class_1

loop_
_rlnOrientationDistribution #1
0.25
0.75
"""
    )
    data = tmp_path / "run_it180_data.star"
    data.write_text("data_particles\n\nloop_\n_rlnImageName #1\n1@stack.mrcs\n")
    sampling = tmp_path / "run_it180_sampling.star"
    sampling.write_text(
        """data_sampling_general
_rlnHealpixOrder 3
_rlnOffsetRange 1.9
_rlnOffsetStep 0.64
_rlnOffsetRangeOriginal 12
_rlnOffsetStepOriginal 4
"""
    )
    optimiser = tmp_path / "run_it180_optimiser.star"
    optimiser.write_text(
        f"""data_optimiser_general
_rlnModelStarFile {model}
_rlnExperimentalDataStarFile {data}
_rlnOrientSamplingStarFile {sampling}
_rlnCurrentIteration 180
_rlnNumberOfIterations 200
_rlnDoGradientRefine 1
_rlnDoStochasticGradientDescent 1
_rlnDoSplitRandomHalves 0
_rlnRandomSeed 29
_rlnAdaptiveOversampleOrder 1
_rlnGradEmIters 0
_rlnParticleDiameter 200
_rlnIncrementImageSize 10
_rlnHasHighFscAtResolLimit 0
_rlnGradCurrentStepsize 0.5
_rlnSgdSubsetSize 1000
_rlnHasConverged 0
_rlnGradHasConverged 0
_rlnAutoLocalSearchesHealpixOrder 4
_rlnOverallAccuracyRotations 0.1
_rlnOverallAccuracyTranslationsAngst 0.4
_rlnChangesOptimalOffsets 1.5
_rlnChangesOptimalOrientations 80
_rlnChangesOptimalClasses 0
_rlnSmallestChangesOffsets 1.5
_rlnSmallestChangesOrientations 80
_rlnSmallestChangesClasses 0
_rlnNumberOfIterWithoutResolutionGain 1
_rlnNumberOfIterWithoutChangingAssignments 0
_rlnBestResolutionThusFar 0.125
"""
    )
    return optimiser, data, {"reference": reference, **moment_values}


def test_native_vdam_diagnostic_continuation_loads_complete_gradient_state(tmp_path):
    optimiser, data, expected = _write_native_vdam_checkpoint(tmp_path)
    opts = driver.NativeInitialModelOptions(
        fn_img=str(data),
        nr_iter=200,
        random_seed=29,
        particle_diameter=200.0,
        diagnostic_continue_optimiser=str(optimiser),
        diagnostic_stop_after_iteration=181,
    )

    checkpoint = driver._load_native_vdam_continuation(
        optimiser,
        expected_data_star=data,
        opts=opts,
        dataset=SimpleNamespace(grid_size=4, voxel_size=2.0),
    )

    from recovar.utils.helpers import relion_volume_to_recovar

    assert checkpoint.iteration == 180
    assert checkpoint.state.iter == 180
    assert checkpoint.state.nr_iter == 200
    assert checkpoint.state.current_resolution_shell == 1
    assert checkpoint.state.subset_size == 1000
    assert checkpoint.state.Igrad1.dtype == np.complex128
    np.testing.assert_array_equal(
        checkpoint.state.Iref[0],
        relion_volume_to_recovar(expected["reference"]),
    )
    np.testing.assert_array_equal(checkpoint.state.Igrad1[0], expected["moment1"])
    np.testing.assert_array_equal(checkpoint.state.Igrad1[1], expected["moment2"])
    np.testing.assert_array_equal(checkpoint.state.Igrad2[0], expected["second_moment"])
    np.testing.assert_array_equal(checkpoint.state.sigma2_noise[0], [0.01, 0.02, 0.03])
    np.testing.assert_array_equal(checkpoint.state.tau2_class[0], [4.0, 5.0, 6.0])
    assert checkpoint.sampling_state.healpix_order == 3
    assert checkpoint.sampling_state.uniform_local_orientation_prior is True


def test_native_vdam_diagnostic_continuation_fails_without_second_pseudo_half(tmp_path):
    optimiser, data, _expected = _write_native_vdam_checkpoint(tmp_path)
    (tmp_path / "run_it180_1moment002.mrc").unlink()
    opts = driver.NativeInitialModelOptions(
        fn_img=str(data),
        nr_iter=200,
        random_seed=29,
        particle_diameter=200.0,
        diagnostic_continue_optimiser=str(optimiser),
        diagnostic_stop_after_iteration=181,
    )

    with pytest.raises(FileNotFoundError):
        driver._load_native_vdam_continuation(
            optimiser,
            expected_data_star=data,
            opts=opts,
            dataset=SimpleNamespace(grid_size=4, voxel_size=2.0),
        )


def test_iteration_reference_replay_expands_iteration_and_class(monkeypatch, tmp_path):
    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.5,
        K=2,
        nr_iter=10,
        n_directions=12,
    )
    paths = []
    expected = []
    for class_index in (1, 2):
        volume = np.full((8, 8, 8), 10.0 + class_index, dtype=np.float64)
        path = tmp_path / f"run_it003_class{class_index:03d}.mrc"
        driver.write_relion_mrc(path, volume, voxel_size=1.5)
        paths.append(path)
        expected.append(volume)
    monkeypatch.setenv(
        driver.INITIAL_MODEL_IREF_REPLAY_TEMPLATE_ENV,
        str(tmp_path / "run_it{iteration:03d}_class{k:03d}.mrc"),
    )
    meta = {}

    replayed = driver._maybe_replay_iteration_references(state, iteration=3, meta=meta)

    np.testing.assert_array_equal(replayed.Iref, np.asarray(expected))
    np.testing.assert_array_equal(state.Iref, 0.0)
    assert meta["diagnostic_iref_replay_paths"] == [str(path) for path in paths]
    assert meta["diagnostic_iref_replay_iteration"] == 3


def test_iteration_reference_replay_rejects_wrong_class_count(monkeypatch):
    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=2,
        nr_iter=10,
        n_directions=12,
    )
    monkeypatch.setenv(driver.INITIAL_MODEL_IREF_REPLAY_TEMPLATE_ENV, "one-map.mrc")

    with pytest.raises(ValueError, match="expects one path for K=1 or K=2"):
        driver._maybe_replay_iteration_references(state, iteration=1, meta={})


def test_experiment_read_order_uses_micrograph_lexicographic_order():
    main = pd.DataFrame(
        {
            "_rlnMicrographName": ["1", "2", "10", "100", "11"],
            "_rlnImageName": ["1@s.mrcs", "2@s.mrcs", "3@s.mrcs", "4@s.mrcs", "5@s.mrcs"],
        }
    )

    assert driver._experiment_read_order(main).tolist() == [0, 2, 3, 4, 1]


def test_seed_zero_halfsets_use_relion_experiment_position_parity():
    main = pd.DataFrame(
        {
            "_rlnMicrographName": ["1", "2", "10", "100", "11"],
            "_rlnImageName": ["1@s.mrcs", "2@s.mrcs", "3@s.mrcs", "4@s.mrcs", "5@s.mrcs"],
        }
    )
    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=1,
        nr_iter=1,
        n_directions=3,
        pseudo_halfsets=True,
    )
    state.subset_size = len(main)

    def fail_if_called(seed):
        raise AssertionError(f"unexpected randomization for seed {seed}")

    out = select_subset_for_iter(
        state,
        iter=1,
        nr_particles=len(main),
        optics_group_by_particle=np.zeros(len(main), dtype=np.int64),
        rnd_unif_factory=fail_if_called,
        random_seed=0,
        do_grad=True,
        particle_order=driver._experiment_read_order(main),
    )

    np.testing.assert_array_equal(out.subset_particle_ids, [0, 2, 3, 4, 1])
    np.testing.assert_array_equal(out.subset_halfset_ids, [0, 1, 0, 1, 0])


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
    main = pd.DataFrame({"_rlnOriginX": ["0.5", "-1.5"], "_rlnOriginY": ["1.6", "-0.49"]})

    shifts = driver._image_pre_shifts_from_star(main, SimpleNamespace(voxel_size=2.0))

    np.testing.assert_array_equal(shifts, np.asarray([[1.0, 2.0], [-2.0, 0.0]], dtype=np.float32))


def test_image_pre_shifts_from_star_rounds_before_float32_downcast():
    main = pd.DataFrame(
        {
            "_rlnOriginX": ["0.49999999", "0.50000001"],
            "_rlnOriginY": ["-0.49999999", "-0.50000001"],
        }
    )

    shifts = driver._image_pre_shifts_from_star(main, SimpleNamespace(voxel_size=2.0))

    np.testing.assert_array_equal(shifts, np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32))


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
    assert state.best_pose_rotations is None


def test_particle_state_from_star_seeds_input_euler_orientations_for_all_particles():
    main = pd.DataFrame(
        {
            "_rlnImageName": ["1@stack.mrcs", "2@stack.mrcs", "3@stack.mrcs"],
            "_rlnAngleRot": [10.0, -75.0, 179.0],
            "_rlnAngleTilt": [35.0, 80.0, 120.0],
            "_rlnAnglePsi": [-20.0, 45.0, 91.0],
        }
    )

    state = driver._particle_state_from_star(main, SimpleNamespace(voxel_size=1.0, n_images=3))

    expected_eulers = main[["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(
        state.best_pose_rotations,
        driver.R_from_relion(expected_eulers, degrees=True).astype(np.float32),
    )
    np.testing.assert_array_equal(state.visited, np.zeros(3, dtype=bool))
    assert driver._best_eulers_from_particle_state(
        state,
        np.asarray([2, 0, 1], dtype=np.int64),
        rotation_grid_order=1,
    ) is not None


def test_sampling_accuracy_uses_seeded_star_eulers_before_particles_are_visited(monkeypatch):
    import recovar.relion_bind as relion_bind

    captured = {}

    def fake_expected_accuracy(*args):
        captured["eulers"] = np.asarray(args[1]).copy()
        captured["particle_ids"] = np.asarray(args[2]).copy()
        return {
            "acc_rot": 2.5,
            "acc_trans": 1.25,
            "acc_rot_class": np.asarray([2.5]),
            "acc_trans_class": np.asarray([1.25]),
            "class_counts": np.asarray([2]),
        }

    monkeypatch.setattr(
        relion_bind,
        "_relion_bind_core",
        SimpleNamespace(vdam_expected_angular_errors=fake_expected_accuracy),
        raising=False,
    )
    main = pd.DataFrame(
        {
            "_rlnImageName": ["1@stack.mrcs", "2@stack.mrcs", "3@stack.mrcs"],
            "_rlnAngleRot": [10.0, -75.0, 179.0],
            "_rlnAngleTilt": [35.0, 80.0, 120.0],
            "_rlnAnglePsi": [-20.0, 45.0, 91.0],
        }
    )
    particle_state = driver._particle_state_from_star(main, SimpleNamespace(voxel_size=2.0, n_images=3))
    state = initialise_denovo_state(ori_size=8, pixel_size=2.0, K=1, nr_iter=200, n_directions=1)
    state.Iref[:] = 1.0
    optics_state = driver.NativeOpticsState(
        voltage=300.0,
        Cs=2.7,
        Q0=0.07,
        pixel_size=2.0,
        defU=np.full(3, 10000.0),
        defV=np.full(3, 10000.0),
        defAngle=np.zeros(3),
        phase_shift=np.zeros(3),
    )

    meta = driver._estimate_native_sampling_accuracy(
        driver._initial_sampling_state(driver.NativeInitialModelOptions(fn_img="particles.star"), pixel_size=2.0),
        state,
        particle_state,
        optics_state,
        particle_order=np.asarray([2, 0], dtype=np.int64),
        random_seed=0,
        padding_factor=1,
        sigma2_fudge=driver.DEFAULT_SIGMA2_FUDGE,
    )

    assert meta is not None
    assert meta["estimated_acc_trans_angstrom"] == 1.25
    np.testing.assert_array_equal(captured["particle_ids"], np.asarray([2, 0], dtype=np.int64))
    np.testing.assert_array_equal(
        captured["eulers"],
        driver.R_to_relion(particle_state.best_pose_rotations[[2, 0]], degrees=True),
    )
    np.testing.assert_array_equal(particle_state.visited, np.zeros(3, dtype=bool))


@pytest.mark.parametrize("missing_name", ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"])
def test_particle_state_from_star_rejects_partial_euler_triplet(missing_name):
    main = pd.DataFrame(
        {
            "_rlnImageName": ["1@stack.mrcs"],
            "_rlnAngleRot": [10.0],
            "_rlnAngleTilt": [35.0],
            "_rlnAnglePsi": [-20.0],
        }
    ).drop(columns=missing_name)

    with pytest.raises(ValueError, match="all Euler-angle columns"):
        driver._particle_state_from_star(main, SimpleNamespace(voxel_size=1.0, n_images=1))


@pytest.mark.parametrize("angle_name", ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"])
def test_particle_state_from_star_rejects_nonfinite_euler_angles(angle_name):
    main = pd.DataFrame(
        {
            "_rlnImageName": ["1@stack.mrcs"],
            "_rlnAngleRot": [10.0],
            "_rlnAngleTilt": [35.0],
            "_rlnAnglePsi": [-20.0],
        }
    )
    main.loc[0, angle_name] = np.nan

    with pytest.raises(ValueError, match="Euler angles must be finite"):
        driver._particle_state_from_star(main, SimpleNamespace(voxel_size=1.0, n_images=1))


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
    assert plan.translations.dtype == np.float32
    assert plan.metadata_translations.shape == plan.translations.shape
    assert plan.metadata_translations.dtype == np.float64
    assert plan.random_perturbation == 0.0


def test_native_expectation_step_uses_rfloat_metadata_translations(monkeypatch):
    metadata_translation = np.float64(1.00000006)

    def fake_build_sampling_plan(opts, *, iteration):
        return driver.NativeSamplingPlan(
            rotations=np.zeros((1, 3, 3), dtype=np.float32),
            translations=np.asarray([[0.0, 0.0], [metadata_translation, 0.0]], dtype=np.float32),
            metadata_translations=np.asarray([[0.0, 0.0], [metadata_translation, 0.0]], dtype=np.float64),
            random_perturbation=0.0,
        )

    def fake_run_dense(dataset, state, config, *, particle_ids, halfset_ids):
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
    particle_state = driver.NativeParticleState(
        translation_offsets=np.zeros((1, 2), dtype=np.float64),
        class_assignments=np.zeros(1, dtype=np.int32),
        max_posterior=np.zeros(1, dtype=np.float32),
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=1)
    state.iter = 1

    expectation_step = driver._native_expectation_step(
        SimpleNamespace(voxel_size=1.0, n_images=1),
        driver.NativeInitialModelOptions(fn_img="particles.star", nr_iter=1),
        np.ones(5, dtype=np.float32),
        particle_state,
    )
    expectation_step(state, np.asarray([0]), np.asarray([0], dtype=np.int8))

    assert particle_state.translation_offsets[0, 0] == metadata_translation
    assert particle_state.translation_offsets[0, 0] != np.float64(np.float32(metadata_translation))


def test_native_driver_rejects_unimplemented_direct_symmetry_before_io():
    opts = driver.NativeInitialModelOptions(
        fn_img="missing.star",
        sym_name="C2",
        do_run_C1=False,
    )

    with pytest.raises(NotImplementedError, match="direct refinement currently supports C1 only"):
        driver.run_native_initial_model(opts)


@pytest.mark.parametrize("chunk_size", [-1, 1, 2])
def test_native_driver_rejects_physical_order_chunks_that_cannot_hold_a_pool_before_io(
    chunk_size,
):
    opts = driver.NativeInitialModelOptions(
        fn_img="missing.star",
        exact_local_physical_order_chunk_size=chunk_size,
    )

    with pytest.raises(ValueError, match="must be 0 .* or at least 3"):
        driver.run_native_initial_model(opts)


def test_native_driver_caches_dataset_immediately_after_loading(monkeypatch):
    events = []
    dataset = SimpleNamespace(tilt_series_flag=False)

    class CacheBoundaryReached(RuntimeError):
        pass

    def fake_load_dataset(*args, **kwargs):
        events.append(("load", args, kwargs))
        return dataset

    def fake_cache(datasets):
        events.append(("cache", tuple(datasets)))
        raise CacheBoundaryReached

    monkeypatch.setattr(driver, "read_star", lambda path: (pd.DataFrame(index=[0]), None))
    monkeypatch.setattr(driver, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(driver, "maybe_cache_raw_image_loaders", fake_cache)
    opts = driver.NativeInitialModelOptions(
        fn_img="particles.star",
        lazy=True,
        datadir="particles",
        strip_prefix="old/",
    )

    with pytest.raises(CacheBoundaryReached):
        driver.run_native_initial_model(opts)

    assert events == [
        (
            "load",
            ("particles.star",),
            {"lazy": True, "datadir": "particles", "strip_prefix": "old/"},
        ),
        ("cache", (dataset,)),
    ]


def test_native_driver_rejects_tilt_series_before_eager_cache(monkeypatch):
    dataset = SimpleNamespace(tilt_series_flag=True)
    monkeypatch.setattr(driver, "read_star", lambda path: (pd.DataFrame(index=[0]), None))
    monkeypatch.setattr(driver, "load_dataset", lambda *args, **kwargs: dataset)

    def fail_if_cached(_datasets):
        raise AssertionError("unsupported tilt-series data must not be eagerly cached")

    monkeypatch.setattr(driver, "maybe_cache_raw_image_loaders", fail_if_cached)

    with pytest.raises(NotImplementedError, match="not tilt-series"):
        driver.run_native_initial_model(driver.NativeInitialModelOptions(fn_img="particles.star"))


class _PersistentRawLoader:
    def __init__(self, *, n=8, D=16):
        self.num_images = n
        self.image_size = D
        self._dtype = np.dtype(np.float32)
        self._cached = None
        self.load_all_count = 0
        self.disk_read_count = 0

    def _read_disk(self, indices):
        self.disk_read_count += 1
        return np.broadcast_to(
            np.asarray(indices, dtype=np.float32)[:, None, None],
            (len(indices), self.image_size, self.image_size),
        ).copy()

    def load_all(self):
        self.load_all_count += 1
        if self._cached is None:
            self._cached = self._read_disk(np.arange(self.num_images))

    def read(self, indices):
        indices = np.asarray(indices, dtype=np.int32)
        if self._cached is not None:
            return self._cached[indices]
        return self._read_disk(indices)


def _raw_cache_dataset(loader):
    return SimpleNamespace(
        image_source=SimpleNamespace(
            backend=SimpleNamespace(source=loader),
        ),
    )


def test_initial_model_raw_cache_persists_across_passes_and_iterations(monkeypatch):
    loader = _PersistentRawLoader()
    monkeypatch.setenv("RECOVAR_EM_RAW_IMAGE_CACHE", "auto")
    monkeypatch.setenv("RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB", "1")

    maybe_cache_raw_image_loaders((_raw_cache_dataset(loader),))
    for _iteration in range(3):
        loader.read([0, 1])  # pass 1
        loader.read([1, 2])  # pass 2

    assert loader.load_all_count == 1
    assert loader.disk_read_count == 1


def test_initial_model_raw_cache_guard_preserves_lazy_loading(monkeypatch):
    loader = _PersistentRawLoader(n=1024, D=1024)
    monkeypatch.setenv("RECOVAR_EM_RAW_IMAGE_CACHE", "auto")
    monkeypatch.setenv("RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB", "0.001")

    maybe_cache_raw_image_loaders((_raw_cache_dataset(loader),))

    assert loader.load_all_count == 0
    assert loader.disk_read_count == 0
    assert loader._cached is None
    loader.read([0])
    assert loader.disk_read_count == 1


def test_configure_relion_image_mask_forwards_image_backend():
    calls = {}

    class Backend:
        def set_relion_image_mask(self, **kwargs):
            calls["mask"] = kwargs

        def set_relion_fourier_backend(self, value):
            calls["fourier_backend"] = value

    dataset = SimpleNamespace(
        image_source=SimpleNamespace(backend=Backend()),
        grid_size=128,
        voxel_size=2.125,
    )
    opts = driver.NativeInitialModelOptions(
        fn_img="particles.star",
        particle_diameter=200.0,
        image_fourier_backend="relion_cuda",
    )

    driver._configure_relion_image_mask(dataset, opts)

    assert calls["mask"] == {
        "pixel_size": 2.125,
        "particle_diameter_ang": 200.0,
        "width_mask_edge_px": 5.0,
    }
    assert calls["fourier_backend"] == "relion_cuda"


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


def test_sampling_plan_applies_relion_radius_tolerance_in_angstroms():
    opts = driver.NativeInitialModelOptions(
        fn_img="particles.star",
        healpix_order=3,
        oversampling=1,
        random_perturbation=0.0,
    )
    sampling_state = driver.NativeSamplingState(
        healpix_order=3,
        adaptive_oversampling=1,
        offset_range_angstrom=6.707714,
        offset_step_angstrom=3.0,
        offset_range_ori_angstrom=25.5,
        offset_step_ori_angstrom=8.5,
        pixel_size=4.25,
    )

    plan = driver._build_sampling_plan(
        opts,
        iteration=20,
        sampling_state=sampling_state,
    )

    assert plan.coarse_prior_translations.shape == (13, 2)
    assert plan.translations.shape == (52, 2)


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
    assert sampling_state.orientational_prior_mode == driver.RELION_ORIENTATIONAL_PRIOR_NOPRIOR
    assert sampling_state.uniform_local_orientation_prior is False
    assert sampling_state.offset_range_angstrom == pytest.approx(12.75)
    assert sampling_state.offset_step_angstrom == pytest.approx(4.25)

    assert driver._prepare_native_sampling_for_iteration(
        sampling_state,
        state,
        iteration=10,
        do_grad=True,
    ) is True
    assert sampling_state.healpix_order == 2
    assert sampling_state.orientational_prior_mode == driver.RELION_ORIENTATIONAL_PRIOR_NOPRIOR
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
    assert sampling_state.orientational_prior_mode == driver.RELION_ORIENTATIONAL_PRIOR_NOPRIOR
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
    assert sampling_state.orientational_prior_mode == driver.RELION_ORIENTATIONAL_PRIOR_ROTTILT_PSI
    assert sampling_state.uniform_local_orientation_prior is True


def test_uniform_local_orientation_prior_replaces_learned_direction_prior():
    state = initialise_denovo_state(ori_size=8, pixel_size=2.0, K=1, nr_iter=200, n_directions=12)
    state.pdf_direction = np.linspace(1.0, 12.0, 12, dtype=np.float64)[None, :]
    sampling_state = driver.NativeSamplingState(
        healpix_order=0,
        adaptive_oversampling=1,
        offset_range_angstrom=12.0,
        offset_step_angstrom=4.0,
        offset_range_ori_angstrom=12.0,
        offset_step_ori_angstrom=4.0,
        pixel_size=2.0,
        orientational_prior_mode=driver.RELION_ORIENTATIONAL_PRIOR_ROTTILT_PSI,
        uniform_local_orientation_prior=True,
    )

    prior = driver._class_rotation_log_prior_for_sampling(state, sampling_state, healpix_order=0)

    assert prior.shape == (1, driver.sampling.rotation_grid_size(0))
    np.testing.assert_array_equal(prior, np.zeros_like(prior))


def test_noprior_sampling_uses_learned_direction_prior(monkeypatch):
    expected = np.asarray([[0.0, 1.0]], dtype=np.float32)
    state = initialise_denovo_state(ori_size=8, pixel_size=2.0, K=1, nr_iter=200, n_directions=12)
    sampling_state = driver.NativeSamplingState(
        healpix_order=0,
        adaptive_oversampling=1,
        offset_range_angstrom=12.0,
        offset_step_angstrom=4.0,
        offset_range_ori_angstrom=12.0,
        offset_step_ori_angstrom=4.0,
        pixel_size=2.0,
    )

    monkeypatch.setattr(driver, "_class_direction_rotation_log_prior", lambda _state, _order: expected)

    prior = driver._class_rotation_log_prior_for_sampling(state, sampling_state, healpix_order=0)

    assert prior is expected


def test_direction_prior_preserves_relion_absolute_log_scale_and_cutoff_tie():
    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=2.0,
        K=1,
        nr_iter=200,
        n_directions=12,
    )
    state.pdf_direction[0, 9] = 0.0194935499409
    state.pdf_direction[0, 10] = 0.0206643770425

    prior = driver._class_direction_rotation_log_prior(state, healpix_order=0)
    n_psi = driver.sampling.rotation_grid_n_in_planes(0)

    expected_9 = np.float32(np.log(state.pdf_direction[0, 9]))
    expected_10 = np.float32(np.log(state.pdf_direction[0, 10]))
    np.testing.assert_array_equal(prior[0, 9 * n_psi : (9 + 1) * n_psi], expected_9)
    np.testing.assert_array_equal(prior[0, 10 * n_psi : (10 + 1) * n_psi], expected_10)

    # Frozen gf10 coarse operands: RELION's absolute log(pdf_direction)
    # makes these rank-21/rank-22 values bitwise tied. Dividing pdf_direction
    # by its mean before log separated them by one float32 ULP and dropped an
    # eight-child fine-rotation parent from the inclusive cutoff support.
    raw_scores = np.asarray([-346.5836181640625, -346.6419677734375], dtype=np.float32)
    tied = raw_scores + np.asarray([expected_9, expected_10], dtype=np.float32)
    assert tied[0].view(np.uint32) == tied[1].view(np.uint32)


def test_active_relion_initialmodel_max_significants_matches_gradient_default():
    state = initialise_denovo_state(ori_size=16, pixel_size=1.0, K=3, nr_iter=8, n_directions=1)

    assert driver._active_relion_initialmodel_max_significants(state, do_grad=True) == 300
    assert driver._active_relion_initialmodel_max_significants(state, do_grad=False) == -1


def test_native_initialmodel_do_grad_honors_terminal_em_iterations():
    state = initialise_denovo_state(ori_size=16, pixel_size=1.0, K=1, nr_iter=8, n_directions=1)

    assert driver._native_initialmodel_do_grad(state, 6, grad_em_iters=2)
    assert not driver._native_initialmodel_do_grad(state, 7, grad_em_iters=2)
    assert not driver._native_initialmodel_do_grad(state, 8, grad_em_iters=2)


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

    seed_zero = driver.NativeInitialModelOptions(
        fn_img="particles.star",
        random_seed=0,
        perturbation_factor=0.5,
    )
    assert driver._random_perturbation_for_iteration(seed_zero, 1) == -0.07990610599517822
    assert driver._random_perturbation_for_iteration(seed_zero, 2) == 0.34533798694610596


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
    np.testing.assert_allclose(
        calls[0]["prior"],
        np.asarray(
            [
                [0.0, -0.025, -0.08],
                [-0.01, -0.065, -0.13],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        calls[1]["prior"],
        np.asarray(
            [
                [-0.025, -0.1, -0.185],
                [-0.13, -0.265, -0.41],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-8,
    )
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

    driver._update_particle_state_from_estep_meta(
        particle_state,
        {
            "selected_particle_ids": np.asarray([1], dtype=np.int64),
            "pose_assignments": np.asarray([0], dtype=np.int32),
        },
        np.asarray([[1.0, -1.0]], dtype=np.float32),
    )

    # RELION writes the latest state for every particle visited by any earlier
    # VDAM subset, not only the identities selected by the current iteration.
    np.testing.assert_array_equal(particle_state.visited, [True, True, True])
    np.testing.assert_allclose(particle_state.best_pose_rotations[[2, 0]], rotations)
    np.testing.assert_array_equal(particle_state.best_pose_rotation_ids, [7, -1, 11])


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
    event_order = []
    prepared_means = np.zeros((1, 8**3), dtype=np.complex64)
    prepared_variance = np.zeros((1, 8**3), dtype=np.float32)
    prepared_half = np.zeros((1, 3, 3, 2), dtype=np.complex64)

    def fake_prepare_projector(state, *, padding_factor):
        event_order.append("prepare_projector")
        assert padding_factor == 2
        return prepared_means, prepared_variance, prepared_half, 2

    def fake_estimate_sampling_accuracy(
        sampling_state,
        state,
        particle_state,
        optics_state,
        *,
        particle_order,
        random_seed,
        padding_factor,
        sigma2_fudge,
    ):
        event_order.append("estimate_accuracy")
        estimate_calls.append(
            {
                "healpix_order": sampling_state.healpix_order,
                "offset_range_angstrom": sampling_state.offset_range_angstrom,
                "particle_order": np.asarray(particle_order, dtype=np.int64).copy(),
                "random_seed": random_seed,
                "padding_factor": padding_factor,
                "sigma2_fudge": sigma2_fudge,
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
        event_order.append("run_estep")
        assert config.engine_kwargs["healpix_order"] == 2
        assert config.translations.shape == (2, 2)
        assert config.means is prepared_means
        assert config.mean_variance is prepared_variance
        assert config.relion_projector_half_by_class is prepared_half
        assert config.relion_projector_r_max == 2
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
    monkeypatch.setattr(driver, "prepare_relion_projector_class_inputs", fake_prepare_projector)
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
    state.tau2_fudge_factor = 3.995253
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
    assert estimate_calls[0]["sigma2_fudge"] == pytest.approx(1.0)
    assert build_calls == [(10, 2, pytest.approx(10.366644))]
    assert meta["sampling_accuracy_estimated"] is True
    assert meta["estimated_acc_rot"] == pytest.approx(3.666)
    assert meta["estimated_acc_trans_angstrom"] == pytest.approx(2.125)
    assert meta["sampling_acc_rot"] == pytest.approx(3.666)
    assert meta["sampling_acc_trans_angstrom"] == pytest.approx(2.125)
    assert meta["offset_range_angstrom"] == pytest.approx(10.366644)
    assert meta["offset_step_angstrom"] == pytest.approx(3.0)
    assert event_order == ["prepare_projector", "estimate_accuracy", "run_estep"]


def test_expected_accuracy_skip_diagnostic_is_explicit_and_strict(monkeypatch):
    monkeypatch.delenv(driver.INITIAL_MODEL_SKIP_EXPECTED_ACCURACY_ENV, raising=False)
    assert driver._skip_native_sampling_accuracy_diagnostic() is False

    monkeypatch.setenv(driver.INITIAL_MODEL_SKIP_EXPECTED_ACCURACY_ENV, "1")
    assert driver._skip_native_sampling_accuracy_diagnostic() is True

    monkeypatch.setenv(driver.INITIAL_MODEL_SKIP_EXPECTED_ACCURACY_ENV, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        driver._skip_native_sampling_accuracy_diagnostic()


def test_expected_accuracy_subprocess_diagnostic_is_explicit_and_strict(monkeypatch):
    monkeypatch.delenv(driver.INITIAL_MODEL_ISOLATE_EXPECTED_ACCURACY_ENV, raising=False)
    assert driver._isolate_native_sampling_accuracy_diagnostic() is False

    monkeypatch.setenv(driver.INITIAL_MODEL_ISOLATE_EXPECTED_ACCURACY_ENV, "1")
    assert driver._isolate_native_sampling_accuracy_diagnostic() is True

    monkeypatch.setenv(driver.INITIAL_MODEL_ISOLATE_EXPECTED_ACCURACY_ENV, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        driver._isolate_native_sampling_accuracy_diagnostic()


def test_sampling_accuracy_binding_uses_sigma2_fudge_not_dynamic_tau2(monkeypatch, tmp_path):
    import recovar.relion_bind as relion_bind

    captured = {}

    def fake_expected_accuracy(*args):
        captured["interpolator"] = args[17]
        captured["sigma2_fudge"] = args[18]
        captured["random_seed_particle_ids"] = np.asarray(args[22]).copy()
        return {
            "acc_rot": 1.823,
            "acc_trans": 1.717,
            "acc_rot_class": np.asarray([1.823]),
            "acc_trans_class": np.asarray([1.717]),
            "class_counts": np.asarray([2]),
        }

    monkeypatch.setattr(
        relion_bind,
        "_relion_bind_core",
        SimpleNamespace(vdam_expected_angular_errors=fake_expected_accuracy),
        raising=False,
    )
    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=2.125,
        K=1,
        nr_iter=200,
        n_directions=1,
    )
    state.Iref[:] = 1.0
    state.iter = 90
    state.tau2_fudge_factor = 3.995253
    state.sorted_particle_ids = np.asarray([1, 0], dtype=np.int64)
    state.sorted_particle_part_ids = np.asarray([9, 4], dtype=np.int64)
    best_rotations = driver.sampling._relion_euler_angles_to_matrix(
        np.asarray([[10.0, 30.0, 20.0], [40.0, 60.0, 50.0]])
    )
    particle_state = driver.NativeParticleState(
        translation_offsets=np.zeros((2, 2), dtype=np.float32),
        class_assignments=np.zeros(2, dtype=np.int32),
        max_posterior=np.ones(2, dtype=np.float32),
        best_pose_rotations=best_rotations,
    )
    optics_state = driver.NativeOpticsState(
        voltage=300.0,
        Cs=2.7,
        Q0=0.07,
        pixel_size=2.125,
        defU=np.full(2, 10000.0),
        defV=np.full(2, 10000.0),
        defAngle=np.zeros(2),
        phase_shift=np.zeros(2),
    )

    monkeypatch.setenv("RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_DUMP_ITERATIONS", "80,90")
    meta = driver._estimate_native_sampling_accuracy(
        driver._initial_sampling_state(
            driver.NativeInitialModelOptions(fn_img="particles.star"),
            pixel_size=2.125,
        ),
        state,
        particle_state,
        optics_state,
        particle_order=np.asarray([1, 0]),
        random_seed=0,
        padding_factor=1,
        sigma2_fudge=driver.DEFAULT_SIGMA2_FUDGE,
    )

    assert captured["sigma2_fudge"] == pytest.approx(1.0)
    assert captured["sigma2_fudge"] != pytest.approx(state.tau2_fudge_factor)
    assert captured["interpolator"] == 1
    np.testing.assert_array_equal(captured["random_seed_particle_ids"], np.asarray([9, 4]))
    assert not np.array_equal(captured["random_seed_particle_ids"], np.asarray([1, 0]))
    assert meta["estimated_acc_sigma2_fudge"] == pytest.approx(1.0)
    np.testing.assert_array_equal(meta["estimated_acc_seed_part_ids"], np.asarray([9, 4]))
    dump = np.load(tmp_path / "iter090_expected_accuracy_inputs.npz")
    assert float(dump["sigma2_fudge"]) == pytest.approx(1.0)
    assert float(dump["acc_rot"]) == pytest.approx(1.823)
    assert float(dump["acc_trans"]) == pytest.approx(1.717)
    np.testing.assert_array_equal(dump["trial_particle_ids"], np.asarray([1, 0]))
    np.testing.assert_array_equal(dump["random_seed_particle_ids"], np.asarray([9, 4]))


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
    dataset = SimpleNamespace(voxel_size=2.0, n_images=1, image_shape=(8, 8))
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


def test_dense_estep_config_propagates_public_pass2_engine():
    dataset = SimpleNamespace(voxel_size=2.0, n_images=1, image_shape=(8, 8))
    opts = driver.NativeInitialModelOptions(
        fn_img="particles.star",
        pass2_engine="compact",
        relion_wavg_sequential_cuda=False,
        exact_local_bucket_radix=2,
        exact_local_physical_order_chunk_size=220,
    )
    plan = driver.NativeSamplingPlan(
        rotations=np.zeros((1, 3, 3), dtype=np.float32),
        translations=np.asarray([[0.0, 0.0]], dtype=np.float32),
        random_perturbation=0.0,
    )

    config = driver._dense_estep_config(
        dataset,
        opts,
        np.ones(5, dtype=np.float32),
        plan,
        np.zeros((1, 2), dtype=np.float32),
    )

    assert config.pass2_engine == "compact"
    assert config.relion_wavg_sequential_cuda is False
    assert config.exact_local_bucket_radix == 2
    assert config.exact_local_physical_order_chunk_size == 220


def test_dense_estep_config_keeps_zero_oversampling_on_exact_adaptive_route():
    dataset = SimpleNamespace(voxel_size=2.0, n_images=1, image_shape=(8, 8))
    opts = driver.NativeInitialModelOptions(fn_img="particles.star", oversampling=0)
    plan = driver.NativeSamplingPlan(
        rotations=np.zeros((72, 3, 3), dtype=np.float32),
        translations=np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        random_perturbation=0.0,
        healpix_order=0,
        oversampling=0,
        offset_step_px=2.0,
        coarse_translations=np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
    )

    config = driver._dense_estep_config(
        dataset,
        opts,
        np.ones(5, dtype=np.float32),
        plan,
        np.zeros((1, 2), dtype=np.float32),
    )

    assert config.engine_kwargs["sparse_pass2"] is True
    assert config.engine_kwargs["oversampling_order"] == 0
    assert config.engine_kwargs["healpix_order"] == 0


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


def test_iteration_zero_artifacts_use_the_normal_iteration_writer(monkeypatch, tmp_path):
    state = initialise_denovo_state(ori_size=8, pixel_size=1.5, K=1, nr_iter=8, n_directions=12)
    main = pd.DataFrame(
        {
            "_rlnImageName": ["1@stack.mrcs", "2@stack.mrcs"],
            "_rlnMicrographName": ["1", "2"],
            "_rlnOpticsGroup": ["1", "1"],
        }
    )
    particle_state = driver.NativeParticleState(
        translation_offsets=np.zeros((2, 2), dtype=np.float32),
        class_assignments=np.zeros(2, dtype=np.int32),
        max_posterior=np.zeros(2, dtype=np.float32),
    )

    def fake_write_mrc(path, volume, *, voxel_size):
        assert np.asarray(volume).shape == (8, 8, 8)
        assert voxel_size == 1.5
        Path(path).write_bytes(b"iteration-zero-map")

    monkeypatch.setattr(driver, "write_relion_mrc", fake_write_mrc)
    prefix = str(tmp_path / "run")
    driver._write_iteration_artifacts(
        prefix,
        state,
        0,
        {"checkpoint_iteration": 0, "phase": "bootstrap"},
        main_star=main,
        optics_star=None,
        dataset=SimpleNamespace(voxel_size=1.5, n_images=2),
        particle_state=particle_state,
    )

    assert (tmp_path / "run_it000_class001.mrc").read_bytes() == b"iteration-zero-map"
    assert (tmp_path / "run_it000_model.star").is_file()
    assert (tmp_path / "run_it000_data.star").is_file()
    meta = json.loads((tmp_path / "run_it000_recovar_meta.json").read_text())
    assert meta == {"checkpoint_iteration": 0, "phase": "bootstrap"}


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
            "--grad_write_iter",
            "7",
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
            "--diagnostic_stop_after_iteration",
            "2",
            "--no_iter_artifacts",
        ]
    )

    assert rc == 0
    opts = calls["opts"]
    assert opts.fn_img == "particles.star"
    assert opts.outputname == "out/run"
    assert opts.nr_iter == 3
    assert opts.grad_write_iter == 7
    assert opts.nr_classes == 2
    assert opts.particle_diameter == 250.0
    assert opts.random_seed == 17
    assert opts.healpix_order == 2
    assert opts.oversampling == 0
    assert opts.offset_range_px == 4.5
    assert opts.offset_step_px == 1.5
    assert opts.random_perturbation == 0.25
    assert opts.translation_sigma_angstrom == 6.5
    assert opts.diagnostic_stop_after_iteration == 2
    assert opts.image_fourier_backend == "host_numpy"
    assert opts.write_iter_artifacts is False
    assert "recovar InitialModel complete: out/initial_model.mrc" in capsys.readouterr().out


def test_cli_gpu_defaults_to_async_relion_cuda_image_backend(monkeypatch):
    run_ab_initio = _load_run_ab_initio()
    calls = {}
    monkeypatch.delenv("CUDA_LAUNCH_BLOCKING", raising=False)

    def fake_run_native(opts):
        calls["opts"] = opts
        return SimpleNamespace(final_mrc="out/initial_model.mrc", final_model_star="out/run_it001_model.star")

    monkeypatch.setattr(driver, "run_native_initial_model", fake_run_native)

    assert run_ab_initio.main(["--i", "particles.star", "--gpu", "0", "--nr_iter", "1"]) == 0
    assert calls["opts"].image_fourier_backend == "relion_cuda"
    assert calls["opts"].deterministic_cuda is False
    assert run_ab_initio.os.environ["CUDA_LAUNCH_BLOCKING"] == "0"


def test_cli_gpu_allows_explicit_deterministic_cuda(monkeypatch):
    run_ab_initio = _load_run_ab_initio()
    calls = {}
    monkeypatch.delenv("CUDA_LAUNCH_BLOCKING", raising=False)

    def fake_run_native(opts):
        calls["opts"] = opts
        return SimpleNamespace(final_mrc="out/initial_model.mrc", final_model_star="out/run_it001_model.star")

    monkeypatch.setattr(driver, "run_native_initial_model", fake_run_native)

    assert (
        run_ab_initio.main(
            ["--i", "particles.star", "--gpu", "0", "--nr_iter", "1", "--deterministic_cuda"]
        )
        == 0
    )
    assert calls["opts"].deterministic_cuda is True
    assert run_ab_initio.os.environ["CUDA_LAUNCH_BLOCKING"] == "1"
