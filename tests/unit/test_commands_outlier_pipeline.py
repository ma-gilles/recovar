import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")

from recovar import utils
from recovar.commands import pipeline_with_outliers
from recovar.commands import run_test_outliers_pipeline


pytestmark = pytest.mark.unit


def test_create_outlier_volume_writes_normalized_mrc(tmp_path):
    out_path = tmp_path / "outlier.mrc"
    run_test_outliers_pipeline.create_outlier_volume(str(out_path), grid_size=16)

    assert out_path.exists()
    vol = utils.load_mrc(str(out_path))
    assert vol.shape == (16, 16, 16)
    assert np.all(np.isfinite(vol))
    assert float(vol.min()) >= -1e-6
    assert float(vol.max()) <= 1.0 + 1e-6


def test_run_test_outliers_pipeline_defaults_to_run_all(monkeypatch, tmp_path):
    commands = []
    removed = []
    os.makedirs(tmp_path / "outliers_test", exist_ok=True)

    def fake_run(command, shell):
        assert shell is True
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_test_outliers_pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(run_test_outliers_pipeline, "create_outlier_volume", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_test_outliers_pipeline, "verify_outlier_results", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        run_test_outliers_pipeline, "analyze_outlier_detection_accuracy", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(run_test_outliers_pipeline, "verify_temp_cleanup", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        run_test_outliers_pipeline.jax,
        "devices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU check should not run with --cpu")),
    )
    monkeypatch.setattr(run_test_outliers_pipeline.shutil, "rmtree", lambda path, **_kwargs: removed.append(path))
    monkeypatch.setattr(
        run_test_outliers_pipeline.sys,
        "argv",
        ["run_test_outliers_pipeline", "--output-dir", str(tmp_path), "--cpu"],
    )

    rc = run_test_outliers_pipeline.main()
    assert rc == 0
    assert any("make_test_dataset" in c for c in commands)
    assert any("pipeline_with_outliers" in c for c in commands)
    assert any("analyze " in c for c in commands)
    assert any("compute_trajectory" in c for c in commands)
    assert removed == [os.path.join(str(tmp_path), "outliers_test")]


def test_run_test_outliers_pipeline_tilt_basic_emits_tilt_flags(monkeypatch, tmp_path):
    commands = []

    monkeypatch.setattr(
        run_test_outliers_pipeline.subprocess,
        "run",
        lambda command, shell: (commands.append(command), SimpleNamespace(returncode=0))[1],
    )
    monkeypatch.setattr(run_test_outliers_pipeline, "create_outlier_volume", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_test_outliers_pipeline, "verify_outlier_results", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        run_test_outliers_pipeline, "analyze_outlier_detection_accuracy", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(run_test_outliers_pipeline, "verify_temp_cleanup", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        run_test_outliers_pipeline.jax,
        "devices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU check should not run with --cpu")),
    )
    monkeypatch.setattr(
        run_test_outliers_pipeline.sys,
        "argv",
        [
            "run_test_outliers_pipeline",
            "--output-dir",
            str(tmp_path),
            "--cpu",
            "--test-basic",
            "--tilt-series",
            "--no-delete",
        ],
    )

    rc = run_test_outliers_pipeline.main()
    assert rc == 0
    assert any("make_test_dataset" in c and "--tilt-series" in c for c in commands)
    assert any("pipeline_with_outliers" in c and "--tilt-series" in c and "particles.star" in c for c in commands)


def test_pipeline_with_outliers_restores_argv_and_uses_previous_round_image_indices(monkeypatch, tmp_path):
    calls = []

    class _FakePO:
        def get(self, key):
            return {4: np.zeros((6, 4), dtype=np.float32)}

        def get_embedding_component(self, entry, zdim):
            return np.zeros((6, 4), dtype=np.float32)

    def fake_standard_recovar_pipeline(args):
        calls.append({"outdir": args.outdir, "ind": args.ind, "tilt_ind": args.tilt_ind})
        # Create per-zdim embedding dir (new format)
        zdim_dir = os.path.join(args.outdir, "model", "zdim_4")
        os.makedirs(zdim_dir, exist_ok=True)
        np.save(os.path.join(zdim_dir, "latent_coords.npy"), np.zeros((6, 4), dtype=np.float32))

    def fake_outlier_main():
        argv = list(pipeline_with_outliers.sys.argv)
        outdir = argv[argv.index("--output-dir") + 1]
        zdim_key = argv[argv.index("--zdim-key") + 1]
        combined = os.path.join(outdir, "data", "combined_results")
        os.makedirs(combined, exist_ok=True)
        with open(os.path.join(combined, f"combined_image_inliers_{zdim_key}.pkl"), "wb") as f:
            pickle.dump(np.array([0, 2, 4], dtype=np.int32), f)
        with open(os.path.join(combined, f"combined_image_outliers_{zdim_key}.pkl"), "wb") as f:
            pickle.dump(np.array([1, 3, 5], dtype=np.int32), f)

    monkeypatch.setattr(pipeline_with_outliers, "standard_recovar_pipeline", fake_standard_recovar_pipeline)
    monkeypatch.setattr(pipeline_with_outliers.output, "PipelineOutput", lambda *_args, **_kwargs: _FakePO())
    monkeypatch.setattr(pipeline_with_outliers.output, "standard_pipeline_plots", lambda *_args, **_kwargs: None)

    import recovar.commands.outlier_detection as outlier_detection_cmd

    monkeypatch.setattr(outlier_detection_cmd, "main", fake_outlier_main)

    original_argv = list(pipeline_with_outliers.sys.argv)
    monkeypatch.setattr(
        pipeline_with_outliers.sys,
        "argv",
        [
            "pipeline_with_outliers",
            str(tmp_path / "particles.64.mrcs"),
            "--poses",
            str(tmp_path / "poses.pkl"),
            "--ctf",
            str(tmp_path / "ctf.pkl"),
            "-o",
            str(tmp_path / "pipeline_out"),
            "--mask",
            "from_halfmaps",
            "--zdim",
            "4",
            "--k-rounds",
            "2",
            "--accept-cpu",
        ],
    )

    pipeline_with_outliers.main()

    assert len(calls) == 2
    assert calls[0]["ind"] is None
    assert calls[1]["ind"].endswith("round_2/inliers_round_1.pkl")
    assert calls[1]["tilt_ind"] is None
    assert list(pipeline_with_outliers.sys.argv) == [
        "pipeline_with_outliers",
        str(tmp_path / "particles.64.mrcs"),
        "--poses",
        str(tmp_path / "poses.pkl"),
        "--ctf",
        str(tmp_path / "ctf.pkl"),
        "-o",
        str(tmp_path / "pipeline_out"),
        "--mask",
        "from_halfmaps",
        "--zdim",
        "4",
        "--k-rounds",
        "2",
        "--accept-cpu",
    ]

    # Restore for hygiene in case later tests introspect process argv.
    monkeypatch.setattr(pipeline_with_outliers.sys, "argv", original_argv)


def test_pipeline_with_outliers_restores_argv_when_outlier_detection_raises(monkeypatch, tmp_path):
    class _FakePO2:
        def get(self, key):
            return {4: np.zeros((4, 4), dtype=np.float32)}

        def get_embedding_component(self, entry, zdim):
            return np.zeros((4, 4), dtype=np.float32)

    def fake_standard_recovar_pipeline(args):
        model_dir = os.path.join(args.outdir, "model")
        zdim_dir = os.path.join(model_dir, "zdim_4")
        os.makedirs(zdim_dir, exist_ok=True)
        np.save(os.path.join(zdim_dir, "latent_coords.npy"), np.zeros((4, 4), dtype=np.float32))

    def fake_outlier_main():
        raise RuntimeError("forced-outlier-failure")

    monkeypatch.setattr(pipeline_with_outliers, "standard_recovar_pipeline", fake_standard_recovar_pipeline)
    monkeypatch.setattr(pipeline_with_outliers.output, "PipelineOutput", lambda *_args, **_kwargs: _FakePO2())
    monkeypatch.setattr(pipeline_with_outliers.output, "standard_pipeline_plots", lambda *_args, **_kwargs: None)

    import recovar.commands.outlier_detection as outlier_detection_cmd

    monkeypatch.setattr(outlier_detection_cmd, "main", fake_outlier_main)

    original_argv = list(pipeline_with_outliers.sys.argv)
    new_argv = [
        "pipeline_with_outliers",
        str(tmp_path / "particles.64.mrcs"),
        "--poses",
        str(tmp_path / "poses.pkl"),
        "--ctf",
        str(tmp_path / "ctf.pkl"),
        "-o",
        str(tmp_path / "pipeline_out"),
        "--mask",
        "from_halfmaps",
        "--zdim",
        "4",
        "--k-rounds",
        "1",
        "--accept-cpu",
    ]
    monkeypatch.setattr(pipeline_with_outliers.sys, "argv", new_argv)

    with pytest.raises(RuntimeError, match="forced-outlier-failure"):
        pipeline_with_outliers.main()

    assert list(pipeline_with_outliers.sys.argv) == new_argv
    monkeypatch.setattr(pipeline_with_outliers.sys, "argv", original_argv)
