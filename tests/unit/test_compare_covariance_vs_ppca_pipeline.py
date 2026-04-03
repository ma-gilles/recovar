import os
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
