import os
from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("jax")

from recovar.ppca import compare_covariance_vs_ppca_pipeline as compare_mod

pytestmark = pytest.mark.unit


def test_run_pipeline_method_adds_ppca_projected_covariance_flag(monkeypatch, tmp_path):
    sim_dir = tmp_path / "simulated_data"
    sim_dir.mkdir()
    (sim_dir / "particles.64.mrcs").write_bytes(b"")
    (sim_dir / "poses.pkl").write_bytes(b"")
    (sim_dir / "ctf.pkl").write_bytes(b"")
    halfsets_path = tmp_path / "halfsets.pkl"
    halfsets_path.write_bytes(b"")
    method_root = tmp_path / "ppca_projected_covariance"

    captured = {}

    def _fake_run(cmd, check, stdout, stderr):
        captured["cmd"] = cmd
        captured["check"] = check
        captured["stdout_name"] = Path(stdout.name).name
        captured["stderr"] = stderr
        return None

    monkeypatch.setattr(compare_mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(compare_mod.time, "time", lambda: 10.0)

    result_dir, log_path, runtime_seconds = compare_mod.run_pipeline_method(
        method="ppca_projected_covariance",
        sim_dir=str(sim_dir),
        method_root=str(method_root),
        grid_size=64,
        halfsets_path=str(halfsets_path),
        zdim=10,
        ppca_em_iters=20,
        use_contrast=True,
        gpu_gb=12,
        low_memory_option=False,
        very_low_memory_option=False,
        lazy=True,
        force=False,
    )

    assert result_dir == os.path.join(str(method_root), "result")
    assert log_path == os.path.join(str(method_root), "ppca_projected_covariance.log")
    assert runtime_seconds == 0.0
    assert captured["check"] is True
    assert captured["stderr"] == compare_mod.subprocess.STDOUT
    assert "--use-ppca" in captured["cmd"]
    assert "--ppca-projected-covariance" in captured["cmd"]
    assert "--correct-contrast" in captured["cmd"]
    assert "--ppca-zdim" in captured["cmd"]
    assert "--ppca-em-iters" in captured["cmd"]


def test_prepare_compare_run_writes_shell_runner(monkeypatch, tmp_path):
    sim_dir = tmp_path / "simulated_data"
    sim_dir.mkdir()
    monkeypatch.setattr(
        compare_mod,
        "_find_cryobench_dataset",
        lambda base_dir, dataset_name: {"name": dataset_name, "vol_dir": str(tmp_path / "vols"), "n_volumes": 3},
    )
    monkeypatch.setattr(compare_mod, "generate_dataset", lambda *args, **kwargs: str(sim_dir))
    monkeypatch.setattr(compare_mod, "ensure_halfsets", lambda *args, **kwargs: str(tmp_path / "halfsets.pkl"))

    args = Namespace(
        base_dir=str(tmp_path),
        results_root=str(tmp_path / "results"),
        dataset="DummySet",
        grid_size=64,
        n_images=1000,
        noise_level=1.0,
        contrast_std=0.3,
        zdim=10,
        ppca_em_iters=20,
        seed=7,
        gpu_gb=12.0,
        covariance_gpu_gb=12.0,
        ppca_gpu_gb=10.0,
        covariance_low_memory_option=True,
        covariance_very_low_memory_option=False,
        ppca_low_memory_option=False,
        ppca_very_low_memory_option=False,
        lazy=True,
        force=False,
    )

    prep = compare_mod.prepare_compare_run(args)

    runner_text = Path(prep["runner_script"]).read_text(encoding="utf-8")
    assert "run_one" in runner_text
    assert "recovar.commands.pipeline" in runner_text
    assert "--ppca-projected-covariance" in runner_text
    assert "skipping covariance; found existing" in runner_text
    assert Path(prep["manifest_path"]).is_file()
    assert len(prep["method_specs"]) == 3
