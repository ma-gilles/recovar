import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.commands import compute_trajectory, make_test_dataset, run_test_dataset

pytestmark = pytest.mark.unit


def test_make_test_dataset_non_tilt_passes_expected_generate_kwargs(monkeypatch, tmp_path):
    calls = {}

    def fake_generate_synthetic_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return object(), {"ok": True}

    monkeypatch.setattr(make_test_dataset.simulator, "generate_synthetic_dataset", fake_generate_synthetic_dataset)

    outdir = tmp_path / "out"
    make_test_dataset.make_test_dataset(str(outdir), image_size=32, noise_level=0.25, n_images=123)

    assert "args" in calls
    assert calls["args"][0].endswith("/test_dataset/")
    assert calls["args"][3] == 123
    assert calls["kwargs"]["grid_size"] == 32
    assert calls["kwargs"]["noise_level"] == 0.25
    assert calls["kwargs"]["disc_type"] == "linear_interp"
    assert "n_tilts" not in calls["kwargs"]


def test_make_test_dataset_tilt_series_passes_tilt_kwargs(monkeypatch, tmp_path):
    calls = {}

    def fake_generate_synthetic_dataset(*args, **kwargs):
        calls["kwargs"] = kwargs
        return object(), {"ok": True}

    monkeypatch.setattr(make_test_dataset.simulator, "generate_synthetic_dataset", fake_generate_synthetic_dataset)

    make_test_dataset.make_test_dataset(
        str(tmp_path / "out"),
        image_size=64,
        noise_level=0.1,
        n_images=270,
        tilt_series=True,
        percent_tilt_series_outliers=0.3,
    )

    assert calls["kwargs"]["n_tilts"] == 27
    assert calls["kwargs"]["dose_per_tilt"] == 3
    assert calls["kwargs"]["angle_per_tilt"] == 3
    assert calls["kwargs"]["percent_tilt_series_outliers"] == 0.3


def test_make_test_dataset_main_parses_and_forwards(monkeypatch, tmp_path):
    captured = {}

    def fake_make_test_dataset(*args, **kwargs):
        captured["args"] = args

    monkeypatch.setattr(make_test_dataset, "make_test_dataset", fake_make_test_dataset)
    monkeypatch.setattr(
        make_test_dataset.sys,
        "argv",
        [
            "make_test_dataset",
            str(tmp_path),
            "--noise-level",
            "0.4",
            "--n-images",
            "77",
            "--image-size",
            "48",
            "--tilt-series",
        ],
    )

    make_test_dataset.main()

    args = captured["args"]
    assert args[0] == str(tmp_path)
    assert args[1] == 48
    assert args[2] == 0.4
    assert args[3] == 77
    assert args[6] is True


def test_compute_trajectory_add_args_parses_ind_list():
    parser = compute_trajectory.add_args(argparse.ArgumentParser())
    args = parser.parse_args(["/tmp/result", "-o", "/tmp/out", "--ind", "3,8"])
    assert args.ind == [3, 8]


def test_compute_trajectory_main_uses_endpts_with_ind(monkeypatch, tmp_path):
    endpts_path = tmp_path / "endpts.txt"
    np.savetxt(endpts_path, np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32))

    captured = {}

    def fake_compute_trajectory(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(compute_trajectory, "compute_trajectory", fake_compute_trajectory)
    monkeypatch.setattr(
        compute_trajectory,
        "copy_data_from_pipeline_output",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        compute_trajectory,
        "cleanup_temp_files",
        lambda *_args, **_kwargs: None,
    )
    parsed_args = SimpleNamespace(
        result_dir=str(tmp_path / "results"),
        outdir=str(tmp_path / "out"),
        zdim=4,
        Bfactor=0,
        n_bins=30,
        n_vols_along_path=6,
        density=None,
        override_z_regularization=False,
        endpts_file=str(endpts_path),
        z_st_file=None,
        z_end_file=None,
        ind=[2, 0],
        copy_to_folder=None,
        no_cleanup=False,
        maskrad_fraction=None,
        n_min_particles=10,
    )

    fake_parser = SimpleNamespace(parse_args=lambda: parsed_args)
    monkeypatch.setattr(compute_trajectory, "add_args", lambda parser: fake_parser)

    compute_trajectory.main()

    assert np.allclose(captured["kwargs"]["z_st"], [4.0, 5.0])
    assert np.allclose(captured["kwargs"]["z_end"], [0.0, 1.0])


def test_compute_trajectory_main_raises_without_endpoints(monkeypatch, tmp_path):
    parsed_args = SimpleNamespace(
        result_dir=str(tmp_path / "results"),
        outdir=str(tmp_path / "out"),
        zdim=4,
        Bfactor=0,
        n_bins=30,
        n_vols_along_path=6,
        density=None,
        override_z_regularization=False,
        endpts_file=None,
        z_st_file=None,
        z_end_file=None,
        ind=None,
        copy_to_folder=None,
        no_cleanup=False,
        maskrad_fraction=None,
        n_min_particles=10,
    )
    fake_parser = SimpleNamespace(parse_args=lambda: parsed_args)
    monkeypatch.setattr(compute_trajectory, "add_args", lambda parser: fake_parser)

    with pytest.raises(Exception, match="end point format wrong"):
        compute_trajectory.main()


def test_run_test_dataset_main_uses_cpu_flag_and_skips_gpu_check(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, shell):
        assert shell is True
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(run_test_dataset.jax, "devices", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU check should not run with --cpu")))
    monkeypatch.setattr(run_test_dataset.os.path, "exists", lambda _p: False)
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(tmp_path), "--cpu"],
    )

    run_test_dataset.main()

    assert commands
    assert all("--accept-cpu" in cmd for cmd in commands if " pipeline " in cmd)


def test_run_test_dataset_tilt_only_emits_tilt_commands(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, shell):
        assert shell is True
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(run_test_dataset.jax, "devices", lambda *_args, **_kwargs: ["gpu0"])
    monkeypatch.setattr(run_test_dataset.os.path, "exists", lambda _p: False)
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(tmp_path), "--tilt-series-only"],
    )

    run_test_dataset.main()

    assert any("make_test_dataset" in cmd and "--tilt-series" in cmd for cmd in commands)
    assert any("pipeline" in cmd and "--tilt-series" in cmd for cmd in commands)
    assert any("reconstruct_from_external_embedding" in cmd and "--tilt-series" in cmd for cmd in commands)
    assert not any("estimate_conformational_density" in cmd for cmd in commands)
