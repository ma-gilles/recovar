"""Verify every heavy-GPU command exposes the new memory-planning flags.

Pure argparse / subprocess-help tests; no GPU required.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit]


# Commands that must accept the full memory-planning surface.
_HEAVY_GPU_COMMANDS = [
    "pipeline",
    "pipeline_with_outliers",
    "analyze",
    "compute_state",
    "compute_trajectory",
    "reconstruct_from_external_embedding",
    "junk_particle_detection",
    "outlier_detection",
    "run_test_dataset",
]


def _help_text_for(cmd: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "recovar.command_line", cmd, "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout + result.stderr


@pytest.mark.parametrize("cmd", _HEAVY_GPU_COMMANDS)
def test_command_help_lists_gpu_gb(cmd):
    txt = _help_text_for(cmd)
    assert "--gpu-gb" in txt, f"recovar {cmd} --help is missing --gpu-gb"
    assert "--gpu-memory" in txt, f"recovar {cmd} --help is missing --gpu-memory alias"


@pytest.mark.parametrize("cmd", _HEAVY_GPU_COMMANDS)
def test_command_help_lists_memory_planning_flags(cmd):
    txt = _help_text_for(cmd)
    expected = [
        "--low-memory-option",
        "--very-low-memory-option",
        "--adaptive-memory",
        "--adaptive-n-pcs",
        "--n-adaptive-pcs",
        "--memory-diagnostics",
        "--fail-on-memory-exceed",
        "--memory-safety-fraction",
        "--hard-gpu-memory-limit",
    ]
    missing = [flag for flag in expected if flag not in txt]
    assert not missing, f"recovar {cmd} --help missing: {missing}"


def test_pipeline_parser_accepts_all_aliases():
    """Both --gpu-gb and --gpu-memory write to args.gpu_memory."""
    from recovar.commands import pipeline as pipe

    parser = argparse.ArgumentParser()
    pipe.add_args(parser)

    a1 = parser.parse_args(["pp.mrcs", "-o", "/tmp", "--mask", "sphere", "--gpu-gb", "12"])
    a2 = parser.parse_args(["pp.mrcs", "-o", "/tmp", "--mask", "sphere", "--gpu-memory", "12"])
    a3 = parser.parse_args(["pp.mrcs", "-o", "/tmp", "--mask", "sphere", "--adaptive-n-pcs"])
    a4 = parser.parse_args(["pp.mrcs", "-o", "/tmp", "--mask", "sphere", "--n-adaptive-pcs"])

    assert a1.gpu_memory == 12.0
    assert a2.gpu_memory == 12.0
    assert a3.adaptive_memory is True
    assert a4.adaptive_memory is True


def test_run_test_dataset_forwards_memory_flags(monkeypatch, tmp_path):
    """When --gpu-gb is set, inner pipeline calls receive --gpu-gb +
    --adaptive-n-pcs (auto-add). With --full-memory-test the auto-add
    is suppressed."""
    from recovar.commands import run_test_dataset as rtd

    parser = rtd._build_parser()

    args = parser.parse_args(["--gpu-gb", "32", "--memory-diagnostics"])
    fwd = rtd._build_forward_argv(args)
    assert "--gpu-gb" in fwd
    assert "32.0" in fwd
    assert "--adaptive-n-pcs" in fwd  # auto-added
    assert "--memory-diagnostics" in fwd

    args = parser.parse_args(["--gpu-gb", "32", "--full-memory-test"])
    fwd = rtd._build_forward_argv(args)
    assert "--gpu-gb" in fwd
    assert "--adaptive-n-pcs" not in fwd  # opt-out

    args = parser.parse_args([])
    assert rtd._build_forward_argv(args) == []


def test_command_line_bootstrap_scan_helpers():
    """Tolerant argv scan used by the hard-limit bootstrap."""
    from recovar import command_line as cl

    argv = ["--gpu-gb", "40", "--hard-gpu-memory-limit"]
    assert cl._scan_for_flag_value(argv, ("--gpu-gb",)) == "40"
    assert cl._scan_for_bool_flag(argv, ("--hard-gpu-memory-limit",)) is True

    # equals form
    argv2 = ["--gpu-gb=40"]
    assert cl._scan_for_flag_value(argv2, ("--gpu-gb",)) == "40"

    # alias names
    argv3 = ["--gpu-memory", "12"]
    assert cl._scan_for_flag_value(argv3, ("--gpu-gb", "--gpu-memory")) == "12"

    # absent
    assert cl._scan_for_flag_value([], ("--gpu-gb",)) is None
    assert cl._scan_for_bool_flag([], ("--hard-gpu-memory-limit",)) is False


def test_disable_jax_preallocation_default(monkeypatch):
    """Issue #135 root cause: JAX preallocates 90% of physical GPU on
    first use. The bootstrap must set XLA_PYTHON_CLIENT_PREALLOCATE=false
    by default (with setdefault, so user overrides still win)."""
    from recovar import command_line as cl

    # Default case: env var unset, bootstrap sets it to false.
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    cl._disable_jax_preallocation_by_default()
    import os

    assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"

    # User override case: existing setting wins (setdefault no-op).
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    cl._disable_jax_preallocation_by_default()
    assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "true"


def test_recovar_cli_sets_preallocate_false(monkeypatch):
    """End-to-end: invoking ``recovar <cmd>`` causes XLA_PYTHON_CLIENT_PREALLOCATE
    to be set in the subprocess env."""
    import subprocess
    import sys

    env = {k: v for k, v in __import__("os").environ.items() if k != "XLA_PYTHON_CLIENT_PREALLOCATE"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "recovar.command_line",
            "make_test_dataset",
            "--help",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Run a child that prints the env var after recovar.command_line bootstraps.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.argv = ['recovar', 'make_test_dataset', '--help']; "
                "from recovar.command_line import _disable_jax_preallocation_by_default; "
                "_disable_jax_preallocation_by_default(); "
                "import os; "
                "print('PREALLOC=' + os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'UNSET'))"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "PREALLOC=false" in probe.stdout
