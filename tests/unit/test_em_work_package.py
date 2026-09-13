"""Bookkeeping must not turn missing observations or mutated source into success."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def package():
    path = Path(__file__).resolve().parents[2] / "scripts/em_work_package.py"
    spec = importlib.util.spec_from_file_location("em_work_package", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("accounting, expected", [
    ("", "UNKNOWN"),
    ("17.batch|COMPLETED|0:0\n", "UNKNOWN"),
    ("17|COMPLETED|0:0\n", "COMPLETED|0:0"),
    ("17|CANCELLED by 42|0:15\n", "CANCELLED|0:15"),
])
def test_missing_queue_requires_exact_accounting_record(package, monkeypatch, accounting, expected):
    monkeypatch.setattr(package.subprocess, "check_output", Mock(side_effect=["", accounting]))
    assert package.job_states(["17"]) == {"17": expected}


def test_requeued_job_uses_live_queue_not_stale_terminal_accounting(package, monkeypatch):
    call = Mock(return_value="17|PENDING\n")
    monkeypatch.setattr(package.subprocess, "check_output", call)
    assert package.job_states(["17"]) == {"17": "PENDING"}
    assert call.call_count == 1


def test_watch_survives_failed_and_missing_observations(package, monkeypatch, tmp_path):
    states = Mock(side_effect=[
        subprocess.TimeoutExpired("squeue", 30),
        {"17": "UNKNOWN"}, {"17": "RUNNING"}, {"17": "FAILED|1:0"},
    ])
    monkeypatch.setattr(package, "job_states", states)
    sleep = Mock()
    monkeypatch.setattr(package.time, "sleep", sleep)
    assert package.watch(SimpleNamespace(jobs=["17"], output=tmp_path, interval=300)) == 0
    assert sleep.call_count == 3
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert all(not event["all_terminal"] for event in events[:-1])
    assert events[-1]["jobs"] == {"17": "FAILED|1:0"}
    assert json.loads((tmp_path / "jobs.json").read_text())["all_terminal"]


@pytest.mark.parametrize("exit_code, changed, expected", [(0, False, 0), (1, False, 1), (0, True, 1)])
def test_command_receipt_fails_on_execution_error_or_source_change(
    package, monkeypatch, tmp_path, exit_code, changed, expected,
):
    before = {"head": "abc", "diff_sha256": "123", "files": {"source.py": "old"}}
    after = {**before, "files": {"source.py": "new"}} if changed else before.copy()
    monkeypatch.setattr(package, "snapshot", Mock(side_effect=[before, after]))
    monkeypatch.setattr(package.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=exit_code)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    args = SimpleNamespace(repo=tmp_path, output=tmp_path, command=["python", "check.py"])
    assert package.run(args) == expected
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["exit_code"] == exit_code
    assert receipt["source_unchanged"] == (not changed)
    assert receipt["command"] == args.command


def test_local_command_requires_explicit_device_visibility(package, monkeypatch, tmp_path):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        package.run(SimpleNamespace(repo=tmp_path, output=tmp_path, command=["python", "check.py"]))


@pytest.mark.parametrize("configured", [False, True])
def test_receipt_records_native_and_cache_overrides_without_unrelated_env(
    package, monkeypatch, tmp_path, configured,
):
    keys = (
        "RECOVAR_DISABLE_CUDA", "RECOVAR_CUDA_LIB", "RECOVAR_CUDA_CACHE_DIR",
        "RECOVAR_RELION_BIND_BUILD_DIR", "RECOVAR_JAX_CACHE_DIR",
        "JAX_COMPILATION_CACHE_DIR",
    )
    expected = {key: ("1" if key == "RECOVAR_DISABLE_CUDA" else str(tmp_path / key))
                if configured else None for key in keys}
    for key, value in expected.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("RECOVAR_TEST_SECRET", "must-not-be-recorded")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(package, "snapshot", lambda repo: {"head": "abc", "diff_sha256": "123"})
    monkeypatch.setattr(package.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0)))
    assert package.run(SimpleNamespace(repo=tmp_path, output=tmp_path, command=["true"])) == 0
    text = (tmp_path / "receipt.json").read_text()
    environment = json.loads(text)["environment"]
    assert {key: environment[key] for key in keys} == expected
    assert "RECOVAR_TEST_SECRET" not in environment
    assert "must-not-be-recorded" not in text
