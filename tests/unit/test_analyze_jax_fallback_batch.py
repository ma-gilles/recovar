"""Tests for the issue #131 follow-up fixes.

Three behaviors:
1. ``--gpu-budget-gb N`` is accepted by analyze / compute_state / compute_trajectory
   and propagates to ``set_gpu_memory_limit()``.
2. Heterogeneity-kernel memory budget is scaled down through the shared
   fallback-path helper when custom CUDA is disabled.
3. ``_lib_is_stale`` detects when the cached ``.so`` was not built from the current
   sources (recorded content hash), triggering a rebuild on next import.
"""

from __future__ import annotations

import argparse
import logging

import pytest


def test_gpu_memory_arg_in_shared_downstream_args():
    """--gpu-budget-gb must be accepted by all downstream commands."""
    from recovar.utils import parser_args

    parser = argparse.ArgumentParser()
    parser_args.standard_downstream_args(parser)
    args = parser.parse_args(["/tmp/results", "--gpu-budget-gb", "8.0"])
    assert args.gpu_memory == 8.0

    # Default is None (auto-detect)
    parser2 = argparse.ArgumentParser()
    parser_args.standard_downstream_args(parser2)
    args2 = parser2.parse_args(["/tmp/results"])
    assert args2.gpu_memory is None


def test_gpu_memory_arg_in_analyze():
    """analyze.py must accept --gpu-budget-gb via standard_downstream_args."""
    from recovar.commands import analyze

    parser = argparse.ArgumentParser()
    analyze.add_args(parser)
    args = parser.parse_args(["--zdim", "4", "--gpu-budget-gb", "12.5", "/dummy/path"])
    assert args.gpu_memory == 12.5


def test_gpu_memory_arg_in_compute_state():
    from recovar.commands import compute_state

    parser = argparse.ArgumentParser()
    compute_state.add_args(parser)
    args = parser.parse_args(["--gpu-budget-gb", "16", "--latent-points", "/dev/null", "/dummy/path"])
    assert args.gpu_memory == 16.0


def test_gpu_memory_arg_in_compute_trajectory():
    from recovar.commands import compute_trajectory

    parser = argparse.ArgumentParser()
    compute_trajectory.add_args(parser)
    args = parser.parse_args(["--gpu-budget-gb", "4", "/dummy/path"])
    assert args.gpu_memory == 4.0


# ---------------------------------------------------------------------------
# --gpu-budget-gb on every other heavy-GPU command
# ---------------------------------------------------------------------------


def _commands_that_need_gpu_memory():
    """Commands that do heavy GPU work and therefore must accept --gpu-budget-gb.

    These are the call sites where the auto-batch-size formula in
    `get_image_batch_size` ultimately fires. If a future contributor adds
    a new heavy-GPU command, they should append it to this list and pull
    `add_gpu_memory_arg` into its parser.
    """
    return [
        "junk_particle_detection",
        "outlier_detection",
        "pipeline_with_outliers",
        "reconstruct_from_external_embedding",
    ]


@pytest.mark.parametrize("cmd_name", _commands_that_need_gpu_memory())
def test_gpu_memory_arg_in_command(cmd_name):
    """Every heavy-GPU command must accept --gpu-budget-gb.

    The check is structural: parse `recovar <cmd> --help` and look for the
    flag string. We don't try to actually invoke the command — many of
    these need a project dir or other args to run end-to-end.
    """
    import subprocess

    from conftest import repo_python_command, repo_subprocess_env

    result = subprocess.run(
        repo_python_command("-m", f"recovar.commands.{cmd_name}", "--help"),
        env=repo_subprocess_env(),
        capture_output=True,
        text=True,
        # Wall-clock guard only: a cold `--help` import takes 15-20 s alone and
        # exceeded 30 s in a 32-way parallel CPU sweep.
        timeout=120,
    )
    # A failed child (including its import-root check) must not pass on text alone.
    assert result.returncode == 0, result.stderr[-2000:]
    # Some commands print help to stdout, some to stderr depending on argparse
    # version + sys.exit code. Check both.
    combined = result.stdout + result.stderr
    assert "--gpu-budget-gb" in combined, (
        f"recovar.commands.{cmd_name} does NOT expose --gpu-budget-gb. "
        f"Pull add_gpu_memory_arg(parser) into its argparse setup so users "
        f"can constrain the auto-batch budget on this command. Without this, "
        f"users on the JAX-fallback path or a smaller GPU will silently "
        f"OOM with no CLI knob to recover.\n"
        f"--- stdout (last 30 lines) ---\n"
        f"{chr(10).join(result.stdout.splitlines()[-30:])}\n"
        f"--- stderr ---\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Heterogeneity-kernel batch-size scaling
# ---------------------------------------------------------------------------


def test_effective_heterogeneity_memory_budget_passthrough(monkeypatch):
    from recovar.heterogeneity import adaptive_kernel_discretization as akd

    monkeypatch.setattr(akd, "custom_cuda_requested", lambda: True)
    assert akd._effective_heterogeneity_memory_budget(48.0) == 48.0


def test_effective_heterogeneity_memory_budget_scales_for_fallback(monkeypatch, caplog):
    from recovar.heterogeneity import adaptive_kernel_discretization as akd

    monkeypatch.setattr(akd, "custom_cuda_requested", lambda: False)
    caplog.set_level(logging.INFO, logger=akd.logger.name)

    scaled = akd._effective_heterogeneity_memory_budget(48.0)

    assert scaled == pytest.approx(4.8)
    assert "scaling heterogeneity-kernel memory budget" in caplog.text


# ---------------------------------------------------------------------------
# Stale .so detection
# ---------------------------------------------------------------------------


def _fake_sources(tmp_path, monkeypatch):
    from recovar import cuda_backproject

    fake_lib_dir = tmp_path / "cuda"
    fake_lib_dir.mkdir()
    (fake_lib_dir / "cuda_backproject.cu").write_text("// source\n")
    (fake_lib_dir / "Makefile").write_text("# makefile\n")
    monkeypatch.setattr(cuda_backproject, "_LIB_DIR", fake_lib_dir)
    monkeypatch.setattr(cuda_backproject, "_lib_missing_required_symbols", lambda _path: None)
    return fake_lib_dir


def test_lib_is_stale_when_sources_changed_since_build(tmp_path, monkeypatch):
    """A library whose recorded source digest no longer matches must be rebuilt."""
    from recovar import cuda_backproject, cuda_build

    fake_lib_dir = _fake_sources(tmp_path, monkeypatch)
    so = tmp_path / "libcuda_backproject.so"
    so.write_bytes(b"build")
    cuda_build.digest_path(so).write_text(cuda_backproject._source_digest())
    assert cuda_backproject._lib_is_stale(so) is False

    (fake_lib_dir / "cuda_backproject.cu").write_text("// changed source\n")
    assert cuda_backproject._lib_is_stale(so) is True


def test_lib_without_recorded_digest_is_stale(tmp_path, monkeypatch):
    from recovar import cuda_backproject

    _fake_sources(tmp_path, monkeypatch)
    so = tmp_path / "libcuda_backproject.so"
    so.write_bytes(b"build of unknown provenance")
    assert cuda_backproject._lib_is_stale(so) is True


def test_lib_is_not_stale_when_identical_sources_are_newer(tmp_path, monkeypatch):
    """A fresh checkout gives identical sources a newer mtime; that alone must not rebuild."""
    import os

    from recovar import cuda_backproject, cuda_build

    fake_lib_dir = _fake_sources(tmp_path, monkeypatch)
    so = tmp_path / "libcuda_backproject.so"
    so.write_bytes(b"build")
    cuda_build.digest_path(so).write_text(cuda_backproject._source_digest())
    old = so.stat().st_mtime - 10_000
    os.utime(so, (old, old))
    (fake_lib_dir / "cuda_backproject.cu").write_text("// source\n")  # same bytes, newer mtime
    assert cuda_backproject._lib_is_stale(so) is False


def test_lib_is_stale_handles_missing_source(tmp_path, monkeypatch):
    """If cuda_backproject.cu doesn't exist (e.g. cuda dir removed),
    _lib_is_stale should not crash and should not falsely report stale."""
    from recovar import cuda_backproject

    fake_lib_dir = tmp_path / "cuda"
    fake_lib_dir.mkdir()
    # No cuda_backproject.cu, no Makefile in fake_lib_dir
    so = tmp_path / "libcuda_backproject.so"
    so.write_bytes(b"build")

    monkeypatch.setattr(cuda_backproject, "_LIB_DIR", fake_lib_dir)
    monkeypatch.setattr(cuda_backproject, "_lib_missing_required_symbols", lambda _path: None)
    # Without sources, can't determine staleness — must default to "not stale"
    # so we don't infinite-rebuild.
    assert cuda_backproject._lib_is_stale(so) is False
