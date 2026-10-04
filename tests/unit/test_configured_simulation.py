"""Explicit simulator configuration must not inherit dataset-specific defaults."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
mrcfile = pytest.importorskip("mrcfile")

from recovar import core  # noqa: E402
from recovar.simulation import configured_simulation as configured  # noqa: E402
from recovar.simulation import simulator  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def config_file(tmp_path):
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    volume[2:5, 3:6, 1:4] = np.arange(27, dtype=np.float32).reshape(3, 3, 3) / 27
    with mrcfile.new(tmp_path / "ground_truth.mrc") as output:
        output.set_data(volume)
        output.voxel_size = 2.25
    config = {
        "volumes": ["ground_truth.mrc"],
        "grid_size": 8,
        "n_particles": 2,
        "n_tilts": 3,
        "angle_per_tilt_deg": 7.0,
        "dose_per_tilt": 1.7,
        "seed": 42,
        "signal_scale": 1.25,
        "poses": {"mode": "uniform"},
        "noise": {"model": "white", "std": 0.3},
        "ctf": {
            "mode": "constant",
            "defocus_u": 12345.0,
            "defocus_v": 12567.0,
            "defocus_angle_deg": 23.0,
            "voltage_kv": 300.0,
            "cs_mm": 2.1,
            "amplitude_contrast": 0.08,
            "phase_shift_deg": 11.0,
            "bfactor": 0.0,
        },
        "tilt_amplitude_weighting": "none",
        "batch_size": 2,
        "noise_rng_batch_size": 4,
    }
    path = tmp_path / "simulation.json"
    path.write_text(json.dumps(config))
    return path, config


def _rewrite(config_file, transform):
    path, original = config_file
    settings = copy.deepcopy(original)
    transform(settings)
    path.write_text(json.dumps(settings))
    return path


@pytest.mark.parametrize(
    "key",
    [
        "volumes",
        "grid_size",
        "n_particles",
        "n_tilts",
        "angle_per_tilt_deg",
        "dose_per_tilt",
        "seed",
        "signal_scale",
        "poses",
        "noise",
        "ctf",
        "tilt_amplitude_weighting",
    ],
)
def test_all_scientific_inputs_are_explicit(config_file, key):
    path = _rewrite(config_file, lambda settings: settings.pop(key))
    with pytest.raises(ValueError, match=key):
        configured.load_simulation_config(path)


@pytest.mark.parametrize("section", [None, "noise", "ctf", "poses"])
def test_unknown_keys_are_not_silently_ignored(config_file, section):
    def mutate(settings):
        (settings if section is None else settings[section])["unexpected_setting"] = 1

    with pytest.raises(ValueError, match="unexpected_setting"):
        configured.load_simulation_config(_rewrite(config_file, mutate))


@pytest.mark.parametrize(
    "key",
    [
        "defocus_u",
        "defocus_v",
        "defocus_angle_deg",
        "voltage_kv",
        "cs_mm",
        "amplitude_contrast",
        "phase_shift_deg",
        "bfactor",
    ],
)
def test_constant_ctf_requires_every_parameter(config_file, key):
    path = _rewrite(config_file, lambda settings: settings["ctf"].pop(key))
    with pytest.raises(ValueError, match=key):
        configured.load_simulation_config(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("grid_size", 7),
        ("grid_size", 0),
        ("n_particles", 0),
        ("n_tilts", 0),
        ("dose_per_tilt", -1),
        ("angle_per_tilt_deg", -1),
        ("signal_scale", 0),
        ("seed", -1),
        ("batch_size", 0),
    ],
)
def test_invalid_numeric_parameters_fail(config_file, key, value):
    path = _rewrite(config_file, lambda settings: settings.update({key: value}))
    with pytest.raises(ValueError):
        configured.load_simulation_config(path)


@pytest.mark.parametrize(
    "noise",
    [
        {"model": "white"},
        {"model": "white", "std": -0.1},
        {"model": "white", "std": float("nan")},
        {"model": "radial"},
        {"model": "radial1"},
    ],
)
def test_noise_does_not_fall_back_to_builtin_template(config_file, noise):
    path = _rewrite(config_file, lambda settings: settings.update(noise=noise))
    with pytest.raises(ValueError):
        configured.load_simulation_config(path)


def test_paths_are_resolved_relative_to_configuration_not_cwd(config_file, tmp_path, monkeypatch):
    path, _ = config_file
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = configured.load_simulation_config(path)
    assert Path(loaded["volumes"][0]) == tmp_path / "ground_truth.mrc"


def test_missing_input_file_is_not_replaced_by_asset(config_file):
    path = _rewrite(config_file, lambda settings: settings.update(volumes=["not_here.mrc"]))
    with pytest.raises((ValueError, FileNotFoundError)):
        configured.load_simulation_config(path)


def _forbid_legacy_defaults(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit configuration must not use legacy dataset defaults")

    for name in [
        "generate_synthetic_dataset",
        "generate_simulated_dataset",
        "get_dataset_params",
        "random_sampling_scheme",
        "get_pose_ctf_generator",
    ]:
        monkeypatch.setattr(simulator, name, forbidden)


def _capture_forward_model(monkeypatch):
    calls = []

    def simulate(
        experiment_dataset,
        volumes,
        noise_variance,
        batch_size,
        image_assignments,
        per_image_contrast,
        per_image_noise_scale,
        **kwargs,
    ):
        calls.append(
            {
                "dataset": experiment_dataset,
                "volumes": np.array(volumes),
                "noise_variance": np.array(noise_variance),
                "image_assignments": np.array(image_assignments),
                "contrast": np.array(per_image_contrast),
                "noise_scale": np.array(per_image_noise_scale),
                "options": kwargs,
            }
        )
        return np.zeros((experiment_dataset.n_images, *experiment_dataset.image_shape), dtype=np.float32)

    monkeypatch.setattr(simulator, "simulate_data", simulate)
    _forbid_legacy_defaults(monkeypatch)
    return calls


def _write_second_volume(tmp_path):
    path = tmp_path / "second_ground_truth.mrc"
    with mrcfile.new(path) as output:
        output.set_data(np.ones((8, 8, 8), dtype=np.float32))
        output.voxel_size = 2.25
    return path.name


def test_volume_assignments_and_weights_are_mutually_exclusive(config_file, tmp_path):
    np.save(tmp_path / "assignments.npy", np.array([0, 0], dtype=np.int64), allow_pickle=False)

    def mutate(settings):
        settings["volume_assignments_file"] = "assignments.npy"
        settings["volume_weights"] = [1.0]

    path = _rewrite(config_file, mutate)
    with pytest.raises(ValueError, match="mutually exclusive"):
        configured.load_simulation_config(path)


@pytest.mark.parametrize(
    "assignments",
    [
        np.array([[0], [0]], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([-1, 0], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([0.0, 0.0], dtype=np.float32),
    ],
)
def test_volume_assignments_reject_invalid_shape_dtype_or_range(config_file, tmp_path, monkeypatch, assignments):
    np.save(tmp_path / "assignments.npy", assignments, allow_pickle=False)
    path = _rewrite(
        config_file,
        lambda settings: settings.update(volume_assignments_file="assignments.npy"),
    )
    calls = _capture_forward_model(monkeypatch)
    config = configured.load_simulation_config(path)
    destination = tmp_path / "invalid_assignments"
    with pytest.raises(ValueError, match="volume_assignments_file"):
        configured.run_configured_simulation(destination, config, output_format="recovar", dry_run=True)
    assert not destination.exists()
    assert calls == []


def test_volume_assignments_are_ordered_repeated_and_idempotently_resolved(config_file, tmp_path, monkeypatch):
    second_volume = _write_second_volume(tmp_path)
    assignments = np.array([1, 0], dtype=np.int64)
    np.save(tmp_path / "assignments.npy", assignments, allow_pickle=False)

    def mutate(settings):
        settings["volumes"].append(second_volume)
        settings["volume_assignments_file"] = "assignments.npy"

    config = configured.load_simulation_config(_rewrite(config_file, mutate))
    assignment_path = str((tmp_path / "assignments.npy").resolve())
    assert config["volume_assignments_file"] == assignment_path
    assert "volume_weights" not in config

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    calls = _capture_forward_model(monkeypatch)
    _, info = configured.run_configured_simulation(tmp_path / "ordered_assignments", config, output_format="recovar")

    expected_images = np.repeat(assignments, 3)
    np.testing.assert_array_equal(info["tilt_series_assignment"], assignments)
    np.testing.assert_array_equal(info["image_assignment"], expected_images)
    np.testing.assert_array_equal(calls[0]["image_assignments"], expected_images)
    resolved = info["simulation_config"]
    assert resolved["volume_assignments_file"] == assignment_path
    assert "volume_weights" not in resolved
    assert resolved["derived"]["volume_assignment_source"] == assignment_path
    assert resolved["derived"]["volume_assignment_counts"] == [1, 1]
    assert "loaded verbatim" in resolved["derived"]["mixture_assignment"]


def test_assignment_replay_does_not_perturb_seeded_pose_or_ctf_sampling(config_file, tmp_path, monkeypatch):
    second_volume = _write_second_volume(tmp_path)
    ctf_rows = np.array(
        [
            [12345.0, 12567.0, 23.0, 300.0, 2.1, 0.08, 11.0, 0.0, 1.0],
            [14321.0, 14678.0, 41.0, 300.0, 2.1, 0.08, 7.0, 0.0, 0.8],
        ],
        dtype=np.float32,
    )
    np.save(tmp_path / "ctf_rows.npy", ctf_rows, allow_pickle=False)

    def sampled_config(settings):
        settings["volumes"].append(second_volume)
        settings["ctf"] = {"mode": "sample", "file": "ctf_rows.npy"}

    calls = _capture_forward_model(monkeypatch)
    sampled = configured.load_simulation_config(_rewrite(config_file, sampled_config))
    first_images, first_info = configured.run_configured_simulation(
        tmp_path / "sampled_assignments", sampled, output_format="recovar"
    )

    np.save(
        tmp_path / "assignments.npy",
        first_info["tilt_series_assignment"],
        allow_pickle=False,
    )

    def replay_config(settings):
        sampled_config(settings)
        settings["volume_assignments_file"] = "assignments.npy"

    replay = configured.load_simulation_config(_rewrite(config_file, replay_config))
    replay_images, replay_info = configured.run_configured_simulation(
        tmp_path / "replayed_assignments", replay, output_format="recovar"
    )

    np.testing.assert_array_equal(replay_images, first_images)
    for key in ("rots", "ctf_params", "tilt_series_assignment", "image_assignment"):
        np.testing.assert_array_equal(replay_info[key], first_info[key])


def _per_tilt_config(config_file, tmp_path, spectrum, variance_scale=1.0):
    np.save(tmp_path / "noise_by_tilt.npy", np.asarray(spectrum), allow_pickle=False)
    path = _rewrite(
        config_file,
        lambda settings: settings.update(
            noise={
                "model": "radial_per_tilt",
                "spectrum_file": "noise_by_tilt.npy",
                "variance_scale": variance_scale,
            }
        ),
    )
    return configured.load_simulation_config(path)


@pytest.mark.parametrize(
    "spectrum",
    [
        np.ones(3),
        np.ones((2, 3)),
        np.ones((3, 4)),
        -np.ones((3, 3)),
        np.full((3, 3), np.nan),
        np.full((3, 3), np.inf),
    ],
)
def test_per_tilt_noise_rejects_invalid_spectrum(config_file, tmp_path, spectrum):
    config = _per_tilt_config(config_file, tmp_path, spectrum)
    destination = tmp_path / "invalid_noise"
    with pytest.raises(ValueError, match="noise.spectrum_file"):
        configured.run_configured_simulation(destination, config, output_format="recovar", dry_run=True)
    assert not destination.exists()


def test_per_tilt_noise_applies_acquisition_rows_and_explicit_seed_children(config_file, tmp_path, monkeypatch):
    spectrum = np.arange(9, dtype=float).reshape(3, 3)
    config = _per_tilt_config(config_file, tmp_path, spectrum, variance_scale=2.5)
    calls = []

    def simulate(dataset, volumes, noise_variance, batch_size, assignments, contrast, noise_scale, **options):
        calls.append((dataset, np.asarray(noise_variance), np.asarray(assignments), options["seed"]))
        doses = np.asarray(dataset.CTF_params)[:, core.CTFParamIndex.DOSE]
        return np.broadcast_to(doses[:, None, None], (dataset.n_images, *dataset.image_shape)).copy()

    monkeypatch.setattr(simulator, "simulate_data", simulate)
    _forbid_legacy_defaults(monkeypatch)
    images, info = configured.run_configured_simulation(tmp_path / "rank_mapping", config, output_format="recovar")
    assert len(calls) == 3
    expected_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in np.random.SeedSequence(config["seed"]).spawn(3)
    ]
    assert info["noise_seeds_per_tilt"] == expected_seeds
    assert info["simulation_config"]["derived"]["noise_seeds_per_tilt"] == expected_seeds
    assert len(set(expected_seeds)) == 3
    for rank, (dataset, powers, assignments, seed) in enumerate(calls):
        rows = np.arange(rank, 6, 3)
        assert dataset.n_images == 2
        np.testing.assert_array_equal(powers, spectrum[rank] * 2.5)
        np.testing.assert_array_equal(dataset.CTF_params, info["ctf_params"][rows])
        np.testing.assert_array_equal(dataset.rotation_matrices, info["rots"][rows])
        np.testing.assert_array_equal(assignments, info["image_assignment"][rows])
        assert seed == expected_seeds[rank]
    np.testing.assert_allclose(images[:, 0, 0], info["ctf_params"][:, core.CTFParamIndex.DOSE])


def test_per_tilt_noise_is_reproducible_and_format_independent(config_file, tmp_path):
    config = _per_tilt_config(config_file, tmp_path, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [8.0, 9.0, 10.0]]))
    first, first_info = configured.run_configured_simulation(tmp_path / "per_tilt_a", config, "recovar")
    again, _ = configured.run_configured_simulation(tmp_path / "per_tilt_b", config, "recovar")
    native, native_info = configured.run_configured_simulation(tmp_path / "per_tilt_native", config, "relion5")
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(first, native)
    np.testing.assert_array_equal(first_info["ctf_params"], native_info["ctf_params"])
    assert first_info["noise_seeds_per_tilt"] == native_info["noise_seeds_per_tilt"]


def test_per_tilt_noise_has_requested_variance_and_independent_particles(config_file, tmp_path):
    with mrcfile.open(tmp_path / "ground_truth.mrc", mode="r+") as handle:
        handle.data[:] = 0
    stds = np.array([0.0, 0.2, 0.5])
    spectrum = np.repeat((stds**2 * 8**2)[:, None], 3, axis=1)
    config = _per_tilt_config(config_file, tmp_path, spectrum)
    config.update(n_particles=30, batch_size=16, noise_rng_batch_size=32)
    images, _ = configured.run_configured_simulation(tmp_path / "per_tilt_measured", config, "recovar")
    np.testing.assert_array_equal(images[0::3], 0)
    for rank in (1, 2):
        assert np.std(images[rank::3]) == pytest.approx(stds[rank], rel=0.1)
        assert not np.array_equal(images[rank], images[rank + 3])


@pytest.mark.parametrize("output_format", ["recovar", "relion5"])
def test_dry_run_creates_no_output_and_does_not_simulate(config_file, tmp_path, monkeypatch, output_format):
    calls = _capture_forward_model(monkeypatch)
    config = configured.load_simulation_config(config_file[0])
    destination = tmp_path / "never_created"
    resolved = configured.run_configured_simulation(destination, config, output_format=output_format, dry_run=True)
    assert isinstance(resolved, dict)
    assert not destination.exists()
    assert calls == []


def test_output_directory_is_never_overwritten(config_file, tmp_path, monkeypatch):
    calls = _capture_forward_model(monkeypatch)
    config = configured.load_simulation_config(config_file[0])
    destination = tmp_path / "occupied"
    destination.mkdir()
    existing = destination / "keep.txt"
    existing.write_text("precious existing output")
    with pytest.raises((ValueError, FileExistsError)):
        configured.run_configured_simulation(destination, config, output_format="recovar")
    assert existing.read_text() == "precious existing output"
    assert calls == []


def test_explicit_parameters_reach_forward_model_without_random_scaling(config_file, tmp_path, monkeypatch):
    calls = _capture_forward_model(monkeypatch)
    config = configured.load_simulation_config(config_file[0])
    images, info = configured.run_configured_simulation(tmp_path / "output", config, output_format="recovar")
    assert len(calls) == 1
    call = calls[0]
    assert images.shape == (6, 8, 8)
    np.testing.assert_allclose(call["noise_variance"], 0.3**2 * 8**2)
    np.testing.assert_array_equal(call["contrast"], np.ones(6))
    np.testing.assert_array_equal(call["noise_scale"], np.ones(6))
    np.testing.assert_array_equal(info["tilt_groups"], [0, 0, 0, 1, 1, 1])
    params = np.asarray(call["dataset"].CTF_params)
    assert params.shape == (6, 11)
    np.testing.assert_allclose(params[:, core.CTFParamIndex.DFU], 12345)
    np.testing.assert_allclose(params[:, core.CTFParamIndex.DFV], 12567)
    np.testing.assert_allclose(params[:, core.CTFParamIndex.VOLT], 300)
    np.testing.assert_allclose(params[:, core.CTFParamIndex.CS], 2.1)
    np.testing.assert_allclose(params[:, core.CTFParamIndex.BFACTOR], 0)
    np.testing.assert_allclose(params[:, core.CTFParamIndex.DOSE], np.tile([0.85, 2.55, 4.25], 2))
    np.testing.assert_array_equal(params[:, core.CTFParamIndex.CONTRAST], np.ones(6))
    assert info["voxel_size"] == pytest.approx(2.25)
    with mrcfile.open(tmp_path / "output" / "particles.8.mrcs") as stack:
        assert float(stack.voxel_size.x) == pytest.approx(2.25)
        np.testing.assert_array_equal(stack.data, images)


def test_radial_noise_uses_only_supplied_spectrum(config_file, tmp_path, monkeypatch):
    spectrum = np.array([3.0, 1.0, 0.5], dtype=np.float32)
    np.save(tmp_path / "noise.npy", spectrum)
    path = _rewrite(
        config_file,
        lambda settings: settings.update(
            noise={"model": "radial", "spectrum_file": "noise.npy", "variance_scale": 2.25}
        ),
    )
    calls = _capture_forward_model(monkeypatch)
    configured.run_configured_simulation(
        tmp_path / "output", configured.load_simulation_config(path), output_format="recovar"
    )
    np.testing.assert_allclose(calls[0]["noise_variance"], spectrum * 2.25)


def test_none_noise_is_explicitly_zero(config_file, tmp_path, monkeypatch):
    path = _rewrite(config_file, lambda settings: settings.update(noise={"model": "none"}))
    calls = _capture_forward_model(monkeypatch)
    configured.run_configured_simulation(
        tmp_path / "output", configured.load_simulation_config(path), output_format="recovar"
    )
    np.testing.assert_array_equal(calls[0]["noise_variance"], 0)


def test_200kv_zero_dose_is_valid_for_native_output(config_file, tmp_path, monkeypatch):
    def mutate(settings):
        settings["ctf"]["voltage_kv"] = 200
        settings["dose_per_tilt"] = 0

    calls = _capture_forward_model(monkeypatch)
    config = configured.load_simulation_config(_rewrite(config_file, mutate))
    configured.run_configured_simulation(tmp_path / "output", config, output_format="relion5", dry_run=True)
    assert not calls


def test_native_nonzero_dose_200kv_fails_before_writing(config_file, tmp_path, monkeypatch):
    path = _rewrite(config_file, lambda settings: settings["ctf"].update(voltage_kv=200))
    calls = _capture_forward_model(monkeypatch)
    with pytest.raises(ValueError):
        configured.run_configured_simulation(
            tmp_path / "output", configured.load_simulation_config(path), output_format="relion5"
        )
    assert not calls
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [("voltage_kv", 0), ("cs_mm", -1), ("amplitude_contrast", -0.1), ("amplitude_contrast", 1.1), ("bfactor", -1)],
)
def test_nonphysical_constant_ctf_parameters_fail(config_file, tmp_path, key, value):
    path = _rewrite(config_file, lambda settings: settings["ctf"].update({key: value}))
    with pytest.raises(ValueError):
        configured.run_configured_simulation(tmp_path / "output", configured.load_simulation_config(path), dry_run=True)
    assert not (tmp_path / "output").exists()


def test_inconsistent_input_voxel_sizes_are_rejected(config_file, tmp_path, monkeypatch):
    with mrcfile.new(tmp_path / "different_voxel.mrc") as output:
        output.set_data(np.ones((8, 8, 8), dtype=np.float32))
        output.voxel_size = 1.0
    path = _rewrite(config_file, lambda settings: settings["volumes"].append("different_voxel.mrc"))
    calls = _capture_forward_model(monkeypatch)
    with pytest.raises(ValueError):
        configured.run_configured_simulation(tmp_path / "output", configured.load_simulation_config(path), dry_run=True)
    assert not calls
    assert not (tmp_path / "output").exists()


def test_nonfinite_volume_is_rejected_during_dry_run(config_file, tmp_path, monkeypatch):
    with mrcfile.open(tmp_path / "ground_truth.mrc", mode="r+") as volume:
        volume.data[0, 0, 0] = np.nan
    calls = _capture_forward_model(monkeypatch)
    with pytest.raises(ValueError):
        configured.run_configured_simulation(
            tmp_path / "output", configured.load_simulation_config(config_file[0]), dry_run=True
        )
    assert not calls
    assert not (tmp_path / "output").exists()


def test_pose_file_with_nonrotation_matrices_is_rejected(config_file, tmp_path, monkeypatch):
    np.save(tmp_path / "poses.npy", np.zeros((2, 3, 3)))
    path = _rewrite(config_file, lambda settings: settings.update(poses={"mode": "file", "file": "poses.npy"}))
    calls = _capture_forward_model(monkeypatch)
    with pytest.raises(ValueError):
        configured.run_configured_simulation(tmp_path / "output", configured.load_simulation_config(path), dry_run=True)
    assert not calls


def test_supplied_base_poses_are_used_without_resampling(config_file, tmp_path, monkeypatch):
    from scipy.spatial.transform import Rotation

    poses = Rotation.from_euler("ZYZ", [[15, 40, 80], [110, 75, -30]], degrees=True).as_matrix()
    np.save(tmp_path / "poses.npy", poses)
    path = _rewrite(config_file, lambda settings: settings.update(poses={"mode": "file", "file": "poses.npy"}))
    _capture_forward_model(monkeypatch)
    _, info = configured.run_configured_simulation(
        tmp_path / "output", configured.load_simulation_config(path), output_format="recovar"
    )
    np.testing.assert_allclose(info["rots"][[0, 3]], poses, rtol=1e-6, atol=1e-6)


def test_negative_radial_power_is_rejected(config_file, tmp_path, monkeypatch):
    np.save(tmp_path / "noise.npy", np.array([1.0, -0.1, 2.0]))
    path = _rewrite(
        config_file,
        lambda settings: settings.update(
            noise={"model": "radial", "spectrum_file": "noise.npy", "variance_scale": 1.0}
        ),
    )
    calls = _capture_forward_model(monkeypatch)
    with pytest.raises(ValueError):
        configured.run_configured_simulation(tmp_path / "output", configured.load_simulation_config(path), dry_run=True)
    assert not calls


def test_explicit_recovar_rejects_unrepresentable_physical_bfactor(config_file, tmp_path, monkeypatch):
    path = _rewrite(config_file, lambda settings: settings["ctf"].update(bfactor=9))
    calls = _capture_forward_model(monkeypatch)
    with pytest.raises(ValueError):
        configured.run_configured_simulation(
            tmp_path / "output", configured.load_simulation_config(path), output_format="recovar"
        )
    assert not calls
    assert not (tmp_path / "output").exists()


def test_tiny_cpu_simulation_is_seeded_reproducibly(config_file, tmp_path, monkeypatch):
    _forbid_legacy_defaults(monkeypatch)
    config = configured.load_simulation_config(config_file[0])
    images_a, info_a = configured.run_configured_simulation(tmp_path / "first", config, output_format="recovar")
    images_b, info_b = configured.run_configured_simulation(tmp_path / "second", config, output_format="recovar")
    np.testing.assert_array_equal(images_a, images_b)
    np.testing.assert_array_equal(info_a["rots"], info_b["rots"])
    changed = copy.deepcopy(config)
    changed["seed"] += 1
    images_c, _ = configured.run_configured_simulation(tmp_path / "third", changed, output_format="recovar")
    assert not np.array_equal(images_a, images_c)


def test_output_format_does_not_change_simulation_physics(config_file, tmp_path, monkeypatch):
    _forbid_legacy_defaults(monkeypatch)
    config = configured.load_simulation_config(config_file[0])
    recovar_images, recovar_info = configured.run_configured_simulation(
        tmp_path / "recovar", config, output_format="recovar"
    )
    native_images, native_info = configured.run_configured_simulation(
        tmp_path / "native", config, output_format="relion5"
    )
    np.testing.assert_array_equal(recovar_images, native_images)
    for key in ["rots", "ctf_params", "noise_variance", "image_assignment"]:
        np.testing.assert_array_equal(recovar_info[key], native_info[key])
    assert (tmp_path / "native" / "optimisation_set.star").is_file()


def test_white_noise_std_matches_real_space_amplitude(config_file, tmp_path, monkeypatch):
    _forbid_legacy_defaults(monkeypatch)
    with mrcfile.new(tmp_path / "zero_signal.mrc") as volume:
        volume.set_data(np.zeros((16, 16, 16), dtype=np.float32))
        volume.voxel_size = 2.25
    path = _rewrite(
        config_file,
        lambda settings: settings.update(
            volumes=["zero_signal.mrc"],
            grid_size=16,
            n_particles=10,
            dose_per_tilt=0,
            batch_size=8,
            noise_rng_batch_size=8,
        ),
    )
    images, _ = configured.run_configured_simulation(
        tmp_path / "output", configured.load_simulation_config(path), output_format="recovar"
    )
    assert images.shape == (30, 16, 16)
    assert float(images.std()) == pytest.approx(0.3, rel=0.1)
    assert abs(float(images.mean())) < 0.02


def _replay_config(config_file, tmp_path):
    from scipy.spatial.transform import Rotation

    rotations = (
        Rotation.from_rotvec(np.arange(18).reshape(6, 3) * 0.037).as_matrix().astype(np.float32).reshape(2, 3, 3, 3)
    )
    translations = (np.arange(12, dtype=np.float32).reshape(2, 3, 2) - 4) / 10
    ctf = np.tile(
        np.array([12345.0, 12567.0, 23.0, 300.0, 2.1, 0.08, 11.0, 0.0, 0.9, 0.0, 0.0], dtype=np.float32), (2, 3, 1)
    )
    ctf[..., 0] += np.arange(6).reshape(2, 3)
    ctf[..., 8] += np.arange(6).reshape(2, 3) * 0.01
    ctf[..., 9] = [[0.0, 2.0, 5.0], [1.0, 3.0, 7.0]]
    np.save(tmp_path / "replay_rotations.npy", rotations, allow_pickle=False)
    np.save(tmp_path / "replay_translations.npy", translations, allow_pickle=False)
    np.save(tmp_path / "replay_ctf.npy", ctf, allow_pickle=False)
    path = _rewrite(
        config_file,
        lambda settings: settings.update(
            poses={
                "mode": "per_image_file",
                "rotations_file": "replay_rotations.npy",
                "translations_file": "replay_translations.npy",
            },
            ctf={"mode": "per_image_file", "file": "replay_ctf.npy"},
            angle_per_tilt_deg=None,
            dose_per_tilt=None,
            tilt_amplitude_weighting="none",
            noise={"model": "none"},
        ),
    )
    return configured.load_simulation_config(path), rotations, translations, ctf


def test_replay_preserves_ordered_metadata_doses_and_star_bfactor(config_file, tmp_path, monkeypatch):
    import starfile

    config, rotations, translations, ctf = _replay_config(config_file, tmp_path)
    calls = _capture_forward_model(monkeypatch)
    _, info = configured.run_configured_simulation(tmp_path / "replay", config, "recovar")
    assert len(calls) == 1
    dataset = calls[0]["dataset"]
    np.testing.assert_array_equal(dataset.rotation_matrices, rotations.reshape(-1, 3, 3))
    np.testing.assert_array_equal(dataset.translations, translations.reshape(-1, 2))
    np.testing.assert_array_equal(dataset.CTF_params, ctf.reshape(-1, 11))
    np.testing.assert_array_equal(info["ctf_params"], ctf.reshape(-1, 11))
    assert info["dose_per_tilt"] is None
    assert info["angle_per_tilt"] is None
    table = starfile.read(tmp_path / "replay" / "particles.star", always_dict=True)["particles"]
    np.testing.assert_array_equal(table["rlnCtfBfactor"], 0)
    np.testing.assert_array_equal(table["rlnMicrographPreExposure"], ctf[..., 9].ravel())
    np.testing.assert_allclose(table["rlnOriginXAngst"], translations[..., 0].ravel() * 2.25, atol=1e-6)
    np.testing.assert_allclose(table["rlnOriginYAngst"], translations[..., 1].ravel() * 2.25, atol=1e-6)


@pytest.mark.parametrize("key", ["dose_per_tilt", "angle_per_tilt_deg"])
def test_replay_requires_null_for_competing_generated_metadata(config_file, tmp_path, key):
    config, _, _, _ = _replay_config(config_file, tmp_path)
    config[key] = 3.0
    with pytest.raises(ValueError, match="must be null"):
        configured.run_configured_simulation(tmp_path / "invalid", config, "recovar", dry_run=True)
    assert not (tmp_path / "invalid").exists()


def test_replay_does_not_silently_add_cosine_weighting(config_file, tmp_path):
    config, _, _, _ = _replay_config(config_file, tmp_path)
    config["tilt_amplitude_weighting"] = "cosine"
    with pytest.raises(ValueError, match="tilt_amplitude_weighting"):
        configured.run_configured_simulation(tmp_path / "invalid", config, "recovar", dry_run=True)


def test_replay_rejects_unsupported_native_export(config_file, tmp_path):
    config, _, _, _ = _replay_config(config_file, tmp_path)
    with pytest.raises(ValueError, match="requires RECOVAR"):
        configured.run_configured_simulation(tmp_path / "invalid", config, "relion5", dry_run=True)


@pytest.mark.parametrize(
    ("column", "values"),
    [
        (9, [[0.0, 3.0, 2.0], [0.0, 1.0, 2.0]]),
        (9, [[-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]]),
        (10, [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
        (7, [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
    ],
)
def test_replay_rejects_unrepresentable_or_invalid_ctf(config_file, tmp_path, column, values):
    config, _, _, ctf = _replay_config(config_file, tmp_path)
    ctf[..., column] = values
    np.save(config["ctf"]["file"], ctf, allow_pickle=False)
    with pytest.raises(ValueError):
        configured.run_configured_simulation(tmp_path / "invalid", config, "recovar", dry_run=True)


@pytest.mark.parametrize("array", ["rotations", "translations", "ctf"])
def test_replay_rejects_malformed_array_shape(config_file, tmp_path, array):
    config, _, _, _ = _replay_config(config_file, tmp_path)
    path = config["ctf"]["file"] if array == "ctf" else config["poses"][f"{array}_file"]
    np.save(path, np.zeros((2, 3)), allow_pickle=False)
    with pytest.raises(ValueError, match="shape"):
        configured.run_configured_simulation(tmp_path / "invalid", config, "recovar", dry_run=True)


def test_replay_matches_direct_forward_model_with_explicit_translations(config_file, tmp_path):
    from recovar.data_io import cryoem_dataset

    config, rotations, translations, ctf = _replay_config(config_file, tmp_path)
    images, _ = configured.run_configured_simulation(tmp_path / "replayed", config, "recovar")
    volumes, voxel_size = simulator.generate_volumes_from_mrcs(config["volumes"], config["grid_size"])
    dataset = cryoem_dataset.CryoEMDataset(
        None,
        voxel_size,
        cryoem_dataset.ImageMetadata(rotations.reshape(-1, 3, 3), translations.reshape(-1, 2), ctf.reshape(-1, 11)),
        ctf_evaluator=core.CTFEvaluator(mode=core.CTFMode.CRYO_ET),
        grid_size=8,
    )
    expected = simulator.simulate_data(
        dataset,
        volumes * config["signal_scale"],
        np.zeros(3),
        config["batch_size"],
        np.zeros(6, dtype=int),
        np.ones(6),
        np.ones(6),
        seed=config["seed"],
        disc_type="linear_interp",
        premultiplied_ctf=False,
        noise_rng_batch_size=config["noise_rng_batch_size"],
    )
    np.testing.assert_array_equal(images, expected)
