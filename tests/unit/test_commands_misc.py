import argparse
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.commands import build_custom_cuda, compute_trajectory, make_test_dataset, run_test_dataset

pytestmark = pytest.mark.unit


def test_build_custom_cuda_main_parses_and_prints(monkeypatch, tmp_path, capsys):
    expected = tmp_path / "libcuda_backproject.so"
    captured = {}

    def fake_build(output_path=None, force=False):
        captured["output_path"] = output_path
        captured["force"] = force
        return expected

    monkeypatch.setattr(build_custom_cuda.cuda_backproject, "build_custom_cuda", fake_build)
    monkeypatch.setattr(sys, "argv", ["build_custom_cuda", "--output", str(expected), "--force"])

    build_custom_cuda.main()

    assert captured == {"output_path": str(expected), "force": True}
    assert capsys.readouterr().out.strip() == str(expected)


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
    assert calls["args"][0].rstrip("/").endswith("/test_dataset")
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


def test_make_test_dataset_tilt_series_respects_explicit_n_tilts_and_volume_input(monkeypatch, tmp_path):
    calls = {}

    def fake_generate_synthetic_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return object(), {"ok": True}

    monkeypatch.setattr(make_test_dataset.simulator, "generate_synthetic_dataset", fake_generate_synthetic_dataset)

    make_test_dataset.make_test_dataset(
        str(tmp_path / "out"),
        image_size=64,
        grid_size=32,
        noise_level=0.1,
        n_images=320,
        tilt_series=True,
        n_tilts=5,
        volume_input=str(tmp_path / "vol"),
    )

    assert calls["args"][2] == str(tmp_path / "vol")
    assert calls["kwargs"]["grid_size"] == 32
    assert calls["kwargs"]["n_tilts"] == 5


def test_make_test_dataset_main_parses_and_forwards(monkeypatch, tmp_path):
    captured = {}

    def fake_make_test_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(make_test_dataset, "make_test_dataset", fake_make_test_dataset)
    monkeypatch.setattr(
        sys,
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
    assert captured["kwargs"]["grid_size"] is None
    assert captured["kwargs"]["volume_input"] is None
    assert captured["kwargs"]["n_tilts"] is None


def test_make_test_dataset_main_parses_grid_volume_and_n_tilts(monkeypatch, tmp_path):
    captured = {}

    def fake_make_test_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(make_test_dataset, "make_test_dataset", fake_make_test_dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_test_dataset",
            str(tmp_path),
            "--grid-size",
            "40",
            "--volume-input",
            str(tmp_path / "vol"),
            "--tilt-series",
            "--n-tilts",
            "9",
        ],
    )

    make_test_dataset.main()
    assert captured["kwargs"]["grid_size"] == 40
    assert captured["kwargs"]["volume_input"] == str(tmp_path / "vol")
    assert captured["kwargs"]["n_tilts"] == 9


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
        maskrad_fraction=None,
        n_min_particles=10,
    )
    fake_parser = SimpleNamespace(parse_args=lambda: parsed_args)
    monkeypatch.setattr(compute_trajectory, "add_args", lambda parser: fake_parser)

    with pytest.raises(Exception, match="end point format wrong"):
        compute_trajectory.main()


def test_compute_trajectory_uses_embedding_component_api_when_available(monkeypatch, tmp_path):
    component_calls = []
    captured = {}

    class _PO:
        def __init__(self, _path):
            self.params = {}

        def get_embedding_keys(self, _entry):
            return [2]

        def get_embedding_component(self, entry, key):
            assert key == 2
            component_calls.append(entry)
            if entry == "latent_coords":
                return np.zeros((5, 2), dtype=np.float64)
            if entry == "latent_precision":
                return np.repeat(np.eye(2, dtype=np.float64)[None, :, :], 5, axis=0)
            if entry == "contrasts":
                return np.ones(5, dtype=np.float64)
            raise KeyError(entry)

        def get(self, key):
            if key in {"latent_coords", "latent_precision", "contrasts"}:
                raise AssertionError(f"compute_trajectory should not call get('{key}') when component API exists")
            if key == "dataset":
                return ["d0"]
            raise KeyError(key)

    monkeypatch.setattr(compute_trajectory.o, "PipelineOutput", _PO)
    monkeypatch.setattr(
        compute_trajectory.embedding,
        "set_contrasts_in_cryos",
        lambda _cryos, contrasts: captured.setdefault("contrast_dtype", np.asarray(contrasts).dtype),
    )
    monkeypatch.setattr(
        compute_trajectory.latent_density,
        "compute_latent_space_density",
        lambda *_args, **_kwargs: (np.ones((4, 4), dtype=np.float32), {"x": [-1, 1]}),
    )
    monkeypatch.setattr(compute_trajectory.o, "mkdir_safe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        compute_trajectory.o,
        "make_trajectory_plots_from_results",
        lambda *_args, **_kwargs: (
            np.zeros((4, 2), dtype=np.float32),
            np.linspace(0.0, 1.0, 3, dtype=np.float32)[:, None].repeat(2, axis=1),
        ),
    )
    monkeypatch.setattr(
        compute_trajectory.o,
        "compute_and_save_reweighted",
        lambda cryos, path, zs, cov_zs, _folder, _bf, _nb, **kwargs: captured.setdefault(
            "call", (cryos, path.shape, zs.shape, cov_zs.shape, kwargs)
        ),
    )

    args = SimpleNamespace(
        Bfactor=0.0,
        n_bins=20,
        maskrad_fraction=0.5,
        n_min_particles=2,
    )
    compute_trajectory.compute_trajectory(
        recovar_result_dir=str(tmp_path / "pipeline_out"),
        output_folder=str(tmp_path / "traj_out"),
        zdim=2,
        B_factor=0.0,
        n_bins=20,
        n_vols_along_path=3,
        density_path=None,
        no_z_reg=False,
        z_st=np.array([0.0, 0.5], dtype=np.float32),
        z_end=np.array([1.0, -0.5], dtype=np.float32),
        args=args,
    )

    assert component_calls.count("latent_coords") == 1
    assert component_calls.count("latent_precision") == 1
    assert component_calls.count("contrasts") == 1
    assert captured["contrast_dtype"] == np.float32
    cryos, path_shape, zs_shape, cov_shape, kwargs = captured["call"]
    assert cryos == ["d0"]
    assert path_shape == (3, 2)
    assert zs_shape == (5, 2)
    assert cov_shape == (5, 2, 2)
    assert kwargs["maskrad_fraction"] == 0.5
    assert kwargs["n_min_particles"] == 2


def test_compute_trajectory_uses_lazy_dataset_when_requested(monkeypatch, tmp_path):
    captured = {"get_keys": []}

    class _PO:
        def __init__(self, _path):
            self.params = {}

        def get_embedding_keys(self, _entry):
            return [2]

        def get_embedding_component(self, entry, _key):
            if entry == "latent_coords":
                return np.zeros((4, 2), dtype=np.float32)
            if entry == "latent_precision":
                return np.repeat(np.eye(2, dtype=np.float32)[None, :, :], 4, axis=0)
            if entry == "contrasts":
                return np.ones(4, dtype=np.float32)
            raise KeyError(entry)

        def get(self, key):
            captured["get_keys"].append(key)
            if key == "lazy_dataset":
                return ["lazy_d0"]
            if key == "dataset":
                return ["dense_d0"]
            raise KeyError(key)

    monkeypatch.setattr(compute_trajectory.o, "PipelineOutput", _PO)
    monkeypatch.setattr(
        compute_trajectory.embedding,
        "set_contrasts_in_cryos",
        lambda cryos, _contrasts: captured.setdefault("cryos", cryos),
    )
    monkeypatch.setattr(
        compute_trajectory.latent_density,
        "compute_latent_space_density",
        lambda *_args, **_kwargs: (np.ones((4, 4), dtype=np.float32), {"x": [-1, 1]}),
    )
    monkeypatch.setattr(compute_trajectory.o, "mkdir_safe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        compute_trajectory.o,
        "make_trajectory_plots_from_results",
        lambda *_args, **_kwargs: (
            np.zeros((4, 2), dtype=np.float32),
            np.linspace(0.0, 1.0, 3, dtype=np.float32)[:, None].repeat(2, axis=1),
        ),
    )
    monkeypatch.setattr(compute_trajectory.o, "compute_and_save_reweighted", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        Bfactor=0.0,
        n_bins=20,
        maskrad_fraction=0.5,
        n_min_particles=2,
        lazy=True,
    )
    compute_trajectory.compute_trajectory(
        recovar_result_dir=str(tmp_path / "pipeline_out"),
        output_folder=str(tmp_path / "traj_out"),
        zdim=2,
        n_vols_along_path=3,
        density_path=None,
        no_z_reg=False,
        z_st=np.array([0.0, 0.5], dtype=np.float32),
        z_end=np.array([1.0, -0.5], dtype=np.float32),
        args=args,
    )

    assert "lazy_dataset" in captured["get_keys"]
    assert "dataset" not in captured["get_keys"]
    assert captured["cryos"] == ["lazy_d0"]


def test_run_test_dataset_main_uses_cpu_flag_and_skips_gpu_check(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **_kwargs):
        # New run_test_dataset uses argv lists + capture_output=True;
        # join into a string so the existing substring assertions still apply.
        commands.append(" ".join(command) if isinstance(command, list) else command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_test_dataset.jax,
        "devices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU check should not run with --cpu")),
    )
    monkeypatch.setattr(run_test_dataset.os.path, "exists", lambda _p: False)
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(tmp_path), "--cpu"],
    )

    run_test_dataset.main()

    assert commands
    # argv lists do not shlex-quote; the executable path appears verbatim.
    assert commands[0].startswith(f"{run_test_dataset.sys.executable} -m recovar.command_line ")
    assert all("--accept-cpu" in cmd for cmd in commands if " pipeline " in cmd)


def test_run_test_dataset_tilt_only_emits_tilt_commands(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **_kwargs):
        # New run_test_dataset uses argv lists + capture_output=True;
        # join into a string so the existing substring assertions still apply.
        commands.append(" ".join(command) if isinstance(command, list) else command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

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


def test_run_test_dataset_deletes_generated_dataset_under_output_dir(monkeypatch, tmp_path):
    commands = []
    removed = []

    def fake_run(command, **_kwargs):
        # New run_test_dataset uses argv lists + capture_output=True;
        # join into a string so the existing substring assertions still apply.
        commands.append(" ".join(command) if isinstance(command, list) else command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    expected = str(tmp_path / "test_dataset")
    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_test_dataset.jax,
        "devices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU check should not run with --cpu")),
    )
    monkeypatch.setattr(run_test_dataset.os.path, "exists", lambda p: p == expected)
    monkeypatch.setattr(run_test_dataset.shutil, "rmtree", lambda p: removed.append(p))
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(tmp_path), "--cpu"],
    )

    run_test_dataset.main()

    assert commands
    assert removed == [expected]


def test_run_test_dataset_tilt_only_deletes_tilt_root_under_output_dir(monkeypatch, tmp_path):
    commands = []
    removed = []

    def fake_run(command, **_kwargs):
        # New run_test_dataset uses argv lists + capture_output=True;
        # join into a string so the existing substring assertions still apply.
        commands.append(" ".join(command) if isinstance(command, list) else command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    expected = str(tmp_path / "tilt_test")
    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(run_test_dataset.jax, "devices", lambda *_args, **_kwargs: ["gpu0"])
    monkeypatch.setattr(run_test_dataset.os.path, "exists", lambda p: p == expected)
    monkeypatch.setattr(run_test_dataset.shutil, "rmtree", lambda p: removed.append(p))
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(tmp_path), "--tilt-series-only"],
    )

    run_test_dataset.main()

    assert commands
    assert removed == [expected]


def test_run_test_dataset_all_tests_emits_extended_commands(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **_kwargs):
        # New run_test_dataset uses argv lists + capture_output=True;
        # join into a string so the existing substring assertions still apply.
        commands.append(" ".join(command) if isinstance(command, list) else command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    # Create per-zdim embedding directory + minimal params.pkl for PipelineOutput
    model_dir = tmp_path / "test_dataset" / "pipeline_output" / "model"
    zdim_dir = model_dir / "zdim_2"
    zdim_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(zdim_dir / "latent_coords.npy"), np.zeros((4, 2), dtype=np.float32))
    with open(model_dir / "params.pkl", "wb") as f:
        pickle.dump({"input_args": SimpleNamespace(zdim=[2])}, f)

    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_test_dataset.jax,
        "devices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU check should not run with --cpu")),
    )
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(tmp_path), "--cpu", "--all-tests", "--no-delete"],
    )

    run_test_dataset.main()

    assert any("pipeline_with_outliers" in cmd for cmd in commands)
    assert any("estimate_stable_states" in cmd for cmd in commands)
    assert any("compute_trajectory" in cmd for cmd in commands)
    assert any("reconstruct_from_external_embedding" in cmd for cmd in commands)


def test_run_test_dataset_all_tests_missing_embeddings_file_skips_reconstruct(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **_kwargs):
        # New run_test_dataset uses argv lists + capture_output=True;
        # join into a string so the existing substring assertions still apply.
        commands.append(" ".join(command) if isinstance(command, list) else command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_test_dataset.jax,
        "devices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU check should not run with --cpu")),
    )
    # No pipeline_output directory → reconstruct step should be skipped.
    # (Don't create any pipeline_output dir in tmp_path)
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(tmp_path), "--cpu", "--all-tests", "--no-delete"],
    )

    # The missing pipeline_output causes prepare_embedding_for_reconstruct
    # to fail, which now triggers a non-zero exit (correct production
    # behavior for any failed function).
    with pytest.raises(SystemExit):
        run_test_dataset.main()

    assert any("pipeline_with_outliers" in cmd for cmd in commands)
    assert any("estimate_stable_states" in cmd for cmd in commands)
    assert not any("reconstruct_from_external_embedding" in cmd for cmd in commands)


def test_run_test_dataset_all_tests_bad_embeddings_payload_skips_reconstruct(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **_kwargs):
        # New run_test_dataset uses argv lists + capture_output=True;
        # join into a string so the existing substring assertions still apply.
        commands.append(" ".join(command) if isinstance(command, list) else command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    embeddings_path = tmp_path / "test_dataset" / "pipeline_output" / "model" / "embeddings.pkl"
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(embeddings_path, "wb") as f:
        # Missing "latent_coords" key should cause reconstruction prep to fail gracefully.
        pickle.dump({"not_latent_coords": {}}, f)

    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_test_dataset.jax,
        "devices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU check should not run with --cpu")),
    )
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(tmp_path), "--cpu", "--all-tests", "--no-delete"],
    )

    # The unparseable embeddings.pkl causes prepare_embedding_for_reconstruct
    # to fail → non-zero exit (correct behavior).
    with pytest.raises(SystemExit):
        run_test_dataset.main()

    assert any("pipeline_with_outliers" in cmd for cmd in commands)
    assert not any("reconstruct_from_external_embedding" in cmd for cmd in commands)


def test_run_test_dataset_no_delete_flag_skips_cleanup(monkeypatch, tmp_path):
    commands = []
    removed = []

    def fake_run(command, **_kwargs):
        # New run_test_dataset uses argv lists + capture_output=True;
        # join into a string so the existing substring assertions still apply.
        commands.append(" ".join(command) if isinstance(command, list) else command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_test_dataset.jax,
        "devices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU check should not run with --cpu")),
    )
    monkeypatch.setattr(run_test_dataset.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(run_test_dataset.shutil, "rmtree", lambda p: removed.append(p))
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(tmp_path), "--cpu", "--no-delete"],
    )

    run_test_dataset.main()

    assert commands
    assert removed == []


def test_run_test_dataset_does_not_cleanup_when_any_command_fails(monkeypatch, tmp_path):
    commands = []
    removed = []

    def fake_run(command, **_kwargs):
        joined = " ".join(command) if isinstance(command, list) else command
        commands.append(joined)
        # Fail one pipeline command to force failed_functions branch.
        if " pipeline " in joined:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_test_dataset.jax,
        "devices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU check should not run with --cpu")),
    )
    monkeypatch.setattr(run_test_dataset.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(run_test_dataset.shutil, "rmtree", lambda p: removed.append(p))
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(tmp_path), "--cpu"],
    )

    # New behavior: exits non-zero when any subprocess failed, so the
    # cleanup-on-failure path is genuinely "no cleanup AND non-zero exit".
    with pytest.raises(SystemExit):
        run_test_dataset.main()

    assert commands
    assert removed == []


def test_run_test_dataset_exits_early_when_no_gpu_and_not_cpu(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(" ".join(command) if isinstance(command, list) else command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(run_test_dataset.jax, "devices", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit):
        run_test_dataset.main()
    assert commands == []


def test_run_test_dataset_handles_paths_with_spaces(monkeypatch, tmp_path):
    """argv lists pass paths verbatim — no shell quoting needed.

    target.txt is now written via Python ``open(..., "w")`` rather than
    a shell redirect (``echo ... > target.txt``); we pre-create the
    parent dir so the writer fires (in production, ``make_test_dataset_tilt``
    creates it) and assert the file lands on disk with the right content.
    """
    commands = []
    outdir = tmp_path / "with space"
    target_parent = outdir / "tilt_test" / "test_dataset"
    target_parent.mkdir(parents=True, exist_ok=True)

    def fake_run(command, **_kwargs):
        commands.append(" ".join(command) if isinstance(command, list) else command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_test_dataset.jax,
        "devices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU check should not run with --cpu")),
    )
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(outdir), "--cpu", "--tilt-series-only", "--no-delete"],
    )

    run_test_dataset.main()

    tilt_root = str(outdir / "tilt_test")
    target = target_parent / "target.txt"
    assert any(f"make_test_dataset {tilt_root} " in cmd for cmd in commands)
    assert target.exists() and target.read_text().strip() == "0.0 0.0"


def test_run_test_dataset_non_tilt_pipeline_paths_with_spaces(monkeypatch, tmp_path):
    """argv lists pass paths verbatim — no shell quoting needed."""
    commands = []
    outdir = tmp_path / "with space"

    def fake_run(command, **_kwargs):
        commands.append(" ".join(command) if isinstance(command, list) else command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_test_dataset.jax,
        "devices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU check should not run with --cpu")),
    )
    monkeypatch.setattr(run_test_dataset.os.path, "exists", lambda _p: False)
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(outdir), "--cpu", "--no-delete"],
    )

    run_test_dataset.main()

    assert commands
    particles = str(outdir / "test_dataset" / "particles.64.mrcs")
    pipeline_out = str(outdir / "test_dataset" / "pipeline_output")
    assert any(f"pipeline {particles} " in cmd for cmd in commands)
    assert any(f"-o {pipeline_out} " in cmd for cmd in commands)


def test_run_test_dataset_custom_synthetic_size_forwarded(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(" ".join(command) if isinstance(command, list) else command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_test_dataset.jax,
        "devices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU check should not run with --cpu")),
    )
    monkeypatch.setattr(run_test_dataset.os.path, "exists", lambda _p: False)
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        [
            "run_test_dataset",
            "--output-dir",
            str(tmp_path),
            "--cpu",
            "--no-delete",
            "--image-size",
            "128",
            "--n-images",
            "2000",
        ],
    )

    run_test_dataset.main()

    assert any("make_test_dataset" in cmd and "--image-size 128" in cmd and "--n-images 2000" in cmd for cmd in commands)
    particles = str(tmp_path / "test_dataset" / "particles.128.mrcs")
    assert any(f"pipeline {particles} " in cmd for cmd in commands)


def test_run_test_dataset_all_tests_quotes_reconstruct_paths_with_spaces(monkeypatch, tmp_path):

    commands = []
    outdir = tmp_path / "with space"
    # Create per-zdim embedding directory + minimal params.pkl for PipelineOutput
    model_dir = outdir / "test_dataset" / "pipeline_output" / "model"
    zdim_dir = model_dir / "zdim_2"
    zdim_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(zdim_dir / "latent_coords.npy"), np.zeros((4, 2), dtype=np.float32))
    with open(model_dir / "params.pkl", "wb") as f:
        pickle.dump({"input_args": SimpleNamespace(zdim=[2])}, f)

    def fake_run(command, **_kwargs):
        # New run_test_dataset uses argv lists + capture_output=True;
        # join into a string so the existing substring assertions still apply.
        commands.append(" ".join(command) if isinstance(command, list) else command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_test_dataset.jax,
        "devices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU check should not run with --cpu")),
    )
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(outdir), "--cpu", "--all-tests", "--no-delete"],
    )

    run_test_dataset.main()

    reconstruct_cmds = [c for c in commands if "reconstruct_from_external_embedding" in c]
    assert reconstruct_cmds
    embedding = str(outdir / "test_dataset" / "embedding_2.pkl")
    target = str(outdir / "test_dataset" / "target.txt")
    assert any(f"--embedding {embedding} " in c for c in reconstruct_cmds)
    assert any(f"--target {target} " in c for c in reconstruct_cmds)


def test_run_test_dataset_skips_wrapper_hint_when_inner_already_printed(monkeypatch, tmp_path, capsys):
    """The inner ``recovar`` subprocess already prints a hint via
    ``command_line._run_with_error_hints``. The wrapper detects the
    delimiter in captured stderr and does NOT print a second hint —
    otherwise the actionable advice gets buried earlier in the Slurm log.
    """
    from recovar.utils import error_hints

    inner_hint = error_hints.format_error_hint(
        error_hints.classify_text(
            "RESOURCE_EXHAUSTED: out of memory",
            error_hints.DiagnosticContext(),
        )
    )
    assert inner_hint.count("TO RECOVER") == 1

    n_calls = {"count": 0}

    def fake_run(command, **_kwargs):
        n_calls["count"] += 1
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=f"some traceback...\n{inner_hint}",
        )

    monkeypatch.setattr(run_test_dataset.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_test_dataset.jax,
        "devices",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no GPU check with --cpu")),
    )
    monkeypatch.setattr(run_test_dataset.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(run_test_dataset.shutil, "rmtree", lambda _p: None)
    monkeypatch.setattr(
        run_test_dataset.sys,
        "argv",
        ["run_test_dataset", "--output-dir", str(tmp_path), "--cpu"],
    )

    with pytest.raises(SystemExit):
        run_test_dataset.main()

    captured = capsys.readouterr()
    # Each failed subprocess echoes its inner hint to stderr exactly once.
    # The wrapper must NOT add a second "TO RECOVER" block per call —
    # otherwise the actionable advice gets buried in the middle of the
    # log instead of staying at the tail.
    n = n_calls["count"]
    assert n > 0, "fake_run was never called — test is broken"
    assert captured.err.count("TO RECOVER") == n, (
        f"expected {n} 'TO RECOVER' (one per failed subprocess), got "
        f"{captured.err.count('TO RECOVER')} — wrapper duplicated the hint."
    )
