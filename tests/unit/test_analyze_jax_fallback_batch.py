"""Tests for the issue #131 follow-up fixes.

Three behaviors:
1. ``--gpu-budget-gb N`` is accepted by analyze / compute_state / compute_trajectory
   and propagates to ``set_gpu_memory_limit()``.
2. Heterogeneity-kernel memory budget is scaled down through the shared
   fallback-path helper when custom CUDA is disabled.
3. ``_lib_is_stale`` detects when the cached ``.so`` is older than its source,
   triggering a rebuild on next import.
"""

from __future__ import annotations

import argparse
import logging
import time

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
    import sys

    result = subprocess.run(
        [sys.executable, "-m", f"recovar.commands.{cmd_name}", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
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


def test_lib_is_stale_when_source_newer(tmp_path, monkeypatch):
    """If cuda_backproject.cu is newer than the .so, _lib_is_stale must
    return True so the auto-build path rebuilds."""
    from recovar import cuda_backproject

    fake_lib_dir = tmp_path / "cuda"
    fake_lib_dir.mkdir()
    src = fake_lib_dir / "cuda_backproject.cu"
    mk = fake_lib_dir / "Makefile"
    so = tmp_path / "libcuda_backproject.so"

    # Write so first (older), then write source (newer).
    so.write_bytes(b"old")
    time.sleep(0.05)
    src.write_text("// source\n")
    mk.write_text("# makefile\n")

    monkeypatch.setattr(cuda_backproject, "_LIB_DIR", fake_lib_dir)
    assert cuda_backproject._lib_is_stale(so) is True


def test_lib_is_not_stale_when_source_older(tmp_path, monkeypatch):
    from recovar import cuda_backproject

    fake_lib_dir = tmp_path / "cuda"
    fake_lib_dir.mkdir()
    src = fake_lib_dir / "cuda_backproject.cu"
    mk = fake_lib_dir / "Makefile"
    so = tmp_path / "libcuda_backproject.so"

    # Source first (older), then .so (newer).
    src.write_text("// source\n")
    mk.write_text("# makefile\n")
    time.sleep(0.05)
    so.write_bytes(b"new build")

    monkeypatch.setattr(cuda_backproject, "_LIB_DIR", fake_lib_dir)
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
    # Without sources, can't determine staleness — must default to "not stale"
    # so we don't infinite-rebuild.
    assert cuda_backproject._lib_is_stale(so) is False
