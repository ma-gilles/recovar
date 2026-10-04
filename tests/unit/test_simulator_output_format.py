"""Output-format selection stays explicit and does not change the simulated images."""

import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy")

from recovar.commands import make_test_dataset
from recovar.simulation import simulator

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("tilt_series", [False, True])
def test_command_default_format_remains_recovar(monkeypatch, tmp_path, tilt_series):
    calls = []

    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return np.empty(0), {}

    monkeypatch.setattr(simulator, "generate_synthetic_dataset", generate)
    make_test_dataset.make_test_dataset(str(tmp_path), n_images=6, tilt_series=tilt_series, n_tilts=3)

    assert len(calls) == 1
    assert calls[0][1]["output_format"] == "recovar"
    assert calls[0][0][3] == 6
    assert "image_dtype" not in calls[0][1]  # Keep the original Python API default.


@pytest.mark.parametrize("output_format", ["recovar", "relion5"])
def test_command_forwards_explicit_configuration(monkeypatch, tmp_path, output_format):
    from recovar.simulation import configured_simulation

    calls = []
    settings = {"explicit": "configuration"}

    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return np.empty(0), {}

    monkeypatch.setattr(configured_simulation, "load_simulation_config", lambda path: settings)
    monkeypatch.setattr(configured_simulation, "run_configured_simulation", generate)
    make_test_dataset.make_test_dataset(
        str(tmp_path),
        simulation_config="explicit.json",
        output_format=output_format,
        dry_run=True,
    )

    assert len(calls) == 1
    assert calls[0][0] == (str(tmp_path / "test_dataset"), settings)
    assert calls[0][1] == {"output_format": output_format, "dry_run": True}
    assert not (tmp_path / "test_dataset").exists()


@pytest.mark.parametrize("output_format", ["recovar", "relion5"])
def test_cli_forwards_explicit_format(monkeypatch, tmp_path, output_format):
    calls = []
    monkeypatch.setattr(make_test_dataset, "make_test_dataset", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_test_dataset", str(tmp_path), "--simulation-config", "explicit.json", "--output-format", output_format],
    )

    make_test_dataset.main()

    assert len(calls) == 1
    assert calls[0][0][0] == str(tmp_path)
    assert calls[0][1]["output_format"] == output_format
    assert calls[0][1]["simulation_config"] == "explicit.json"


@pytest.mark.parametrize(
    "options",
    [
        ["--output-format", "relion5"],
        ["--dry-run"],
        ["--simulation-config", "explicit.json", "--noise-level", "0.1"],
        ["--simulation-config", "explicit.json", "--seed=42"],
        ["--simulation-config", "explicit.json", "--tilt-series"],
    ],
)
def test_cli_rejects_ambiguous_or_incomplete_configuration(monkeypatch, options):
    def unexpected_generation(*args, **kwargs):
        pytest.fail("Ambiguous configuration must fail before generation")

    monkeypatch.setattr(make_test_dataset, "make_test_dataset", unexpected_generation)
    monkeypatch.setattr(sys, "argv", ["make_test_dataset", *options])
    with pytest.raises(SystemExit) as exc:
        make_test_dataset.main()
    assert exc.value.code == 2


def test_cli_default_format_is_recovar(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(make_test_dataset, "make_test_dataset", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["make_test_dataset", str(tmp_path)])

    make_test_dataset.main()

    assert calls[0]["output_format"] == "recovar"


def test_cli_rejects_unknown_format(monkeypatch, capsys):
    def unexpected_generation(*args, **kwargs):
        pytest.fail("An invalid output format must be rejected before generation")

    monkeypatch.setattr(make_test_dataset, "make_test_dataset", unexpected_generation)
    monkeypatch.setattr(sys, "argv", ["make_test_dataset", "--output-format", "relion4"])

    with pytest.raises(SystemExit) as exc:
        make_test_dataset.main()

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_format": "invalid"},
        {"output_format": "relion5"},
        {"output_format": "relion5", "n_tilts": 0},
        {"output_format": "relion5", "n_tilts": 3, "premultiplied_ctf": True},
        {"output_format": "relion5", "n_tilts": 3, "create_nested_structure": True},
    ],
)
def test_generator_rejects_invalid_format_before_loading_or_creating_output(monkeypatch, tmp_path, kwargs):
    output_dir = tmp_path / "must_not_be_created"

    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid format combinations must fail before reading input volumes")

    monkeypatch.setattr(simulator, "load_volumes_from_folder", unexpected_load)

    with pytest.raises(ValueError):
        simulator.generate_synthetic_dataset(str(output_dir), 1.0, "nonexistent_vol", 6, **kwargs)

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_format": "invalid"},
        {"output_format": "relion5"},
        {"output_format": "relion5", "tilt_series": True, "n_tilts": 0},
        {"output_format": "relion5", "tilt_series": True, "premultiplied_ctf": True},
        {"output_format": "relion5", "tilt_series": True, "create_nested_structure": True},
    ],
)
def test_command_rejects_invalid_format_before_creating_output(monkeypatch, tmp_path, kwargs):
    output_dir = tmp_path / "must_not_be_created"

    def unexpected_generation(*args, **kw):
        pytest.fail("Invalid format combinations must fail before simulator dispatch")

    monkeypatch.setattr(simulator, "generate_synthetic_dataset", unexpected_generation)

    with pytest.raises(ValueError):
        make_test_dataset.make_test_dataset(str(output_dir), **kwargs)

    assert not output_dir.exists()


@pytest.mark.parametrize("n_images", [0, -3, 5])
def test_legacy_native_request_requires_configuration_before_loading(monkeypatch, tmp_path, n_images):
    output_dir = tmp_path / "must_not_be_created"

    def unexpected_load(*args, **kwargs):
        pytest.fail("Legacy native-output requests must fail before reading input volumes")

    monkeypatch.setattr(simulator, "load_volumes_from_folder", unexpected_load)

    with pytest.raises(ValueError, match="requires explicit configuration"):
        simulator.generate_synthetic_dataset(
            output_dir, 1.0, "nonexistent_vol", n_images, n_tilts=3, output_format="relion5"
        )

    assert not output_dir.exists()


def test_legacy_native_request_preserves_existing_results(monkeypatch, tmp_path):
    sentinel = tmp_path / "existing_results.txt"
    sentinel.write_text("Keep existing scientific results unchanged.\n")

    def unexpected_load(*args, **kwargs):
        pytest.fail("Legacy native-output requests must fail before reading input volumes")

    monkeypatch.setattr(simulator, "load_volumes_from_folder", unexpected_load)

    with pytest.raises(ValueError, match="requires explicit configuration"):
        simulator.generate_synthetic_dataset(tmp_path, 1.0, "nonexistent_vol", 6, n_tilts=3, output_format="relion5")

    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == "Keep existing scientific results unchanged.\n"


@pytest.fixture
def fake_simulation(monkeypatch):
    """Exercise real output dispatch, but not projection/noise generation or GPU code."""
    calls = []

    def parameters(n_images, grid_size):
        ctf = np.zeros((n_images, 11), dtype=np.float32)
        rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (n_images, 3, 3)).copy()
        translations = np.zeros((n_images, 2), dtype=np.float32)
        return ctf, rotations, translations

    def generate(*args, **kwargs):
        n_images = args[3]
        images = np.ones((n_images, 4, 4), dtype=np.float32)
        images[:, 0, 0] = np.arange(n_images, dtype=np.float32)
        ctf, rotations, translations = kwargs["dataset_param_generator"](n_images, 4)
        n_tilts = kwargs.get("n_tilts", -1)
        groups = np.arange(n_images) // n_tilts if n_tilts > 0 else None
        if n_tilts > 0:
            dose = ((np.arange(n_images) % n_tilts) + 0.5) * kwargs["dose_per_tilt"]
            ctf = np.concatenate([ctf, dose[:, None], np.zeros((n_images, 1))], axis=1)
        info = {"n_tilts": n_tilts, "dose_per_tilt": 3, "angle_per_tilt": 3}
        result = images, ctf, rotations, translations, info, args[1], groups
        calls.append(result)
        return result

    monkeypatch.setattr(simulator, "load_volumes_from_folder", lambda *args, **kw: np.ones((1, 4**3)))
    monkeypatch.setattr(simulator, "get_pose_ctf_generator", lambda *args: parameters)
    monkeypatch.setattr(simulator, "get_noise_model", lambda *args: np.ones(1))
    monkeypatch.setattr(simulator, "generate_simulated_dataset", generate)
    return calls


def test_generator_default_keeps_legacy_recovar_outputs(monkeypatch, tmp_path, fake_simulation):
    star_calls = []
    monkeypatch.setattr(simulator.utils, "write_starfile", lambda *args, **kwargs: star_calls.append((args, kwargs)))

    images, info = simulator.generate_synthetic_dataset(str(tmp_path), 1.5, "fake_vol", 6, grid_size=4, n_tilts=3)

    assert "output_format" not in info
    assert "relion5_export" not in info
    assert "forward_ctf_bfactor" not in info
    assert set(info) == {
        "n_tilts",
        "dose_per_tilt",
        "angle_per_tilt",
        "volumes_path_root",
        "trailing_zero_format_in_vol_name",
        "scale_vol",
        "grid_size",
        "dataset_params_option",
        "outlier_file_input",
        "noise_model",
        "noise_level",
    }
    assert len(star_calls) == 1
    assert Path(star_calls[0][0][6]) == tmp_path / "particles.star"
    assert (tmp_path / "poses.pkl").is_file()
    assert (tmp_path / "ctf.pkl").is_file()
    assert (tmp_path / "simulation_info.pkl").is_file()
    with mrcfile.open(tmp_path / "particles.4.mrcs") as stack:
        np.testing.assert_array_equal(stack.data, images)
        assert stack.data.dtype == np.float16
    assert images is fake_simulation[-1][0]
    # Default RECOVAR generation receives the original 11-column generator,
    # unchanged; its legacy tilt branch appends two columns.
    assert fake_simulation[-1][1].shape == (6, 13)


def test_legacy_generator_cannot_silently_supply_native_defaults(tmp_path, fake_simulation):
    with pytest.raises(ValueError, match="requires explicit configuration"):
        simulator.generate_synthetic_dataset(
            str(tmp_path), 1.5, "fake_vol", 6, grid_size=4, n_tilts=3, output_format="relion5"
        )
    assert not fake_simulation
    assert not list(tmp_path.iterdir())
