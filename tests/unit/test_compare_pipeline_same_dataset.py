import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_compare_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "compare_pipeline_same_dataset.py"
    spec = importlib.util.spec_from_file_location("compare_pipeline_same_dataset", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sanitize_pipeline_extra_args_drops_very_low_memory_option():
    mod = _load_compare_module()
    args = ["--mask", "from_halfmaps", "--very-low-memory-option", "--gpu-gb", "20", "--correct-contrast"]
    out = mod.sanitize_pipeline_extra_args(args)
    assert "--very-low-memory-option" not in out
    # Keep unrelated args untouched.
    assert out == ["--mask", "from_halfmaps", "--gpu-gb", "20", "--correct-contrast"]


def test_main_strips_very_low_memory_from_pipeline_commands(monkeypatch, tmp_path):
    mod = _load_compare_module()
    ds = tmp_path / "dataset"
    ds.mkdir()
    (ds / "particles.64.mrcs").write_bytes(b"")
    (ds / "poses.pkl").write_bytes(b"")
    (ds / "ctf.pkl").write_bytes(b"")
    current = tmp_path / "current_repo"
    other = tmp_path / "other_repo"
    current.mkdir()
    other.mkdir()

    calls = []

    def fake_run(cmd, check, env, cwd):
        calls.append((list(cmd), cwd))
        return 0

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    argv = [
        "compare_pipeline_same_dataset.py",
        "--dataset-dir",
        str(ds),
        "--current-repo-root",
        str(current),
        "--other-repo-root",
        str(other),
        "--output-base",
        str(tmp_path / "out"),
        "--pipeline-extra-args",
        "--mask from_halfmaps --noise-model radial --correct-contrast --very-low-memory-option --gpu-gb 20",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    mod.main()

    pipeline_cmds = [c[0] for c in calls if len(c[0]) >= 5 and c[0][3] == "pipeline"]
    assert len(pipeline_cmds) == 2
    for cmd in pipeline_cmds:
        assert "--very-low-memory-option" not in cmd
        assert "--gpu-gb" in cmd
