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
    assert "--gpu-budget-gb" in txt, f"recovar {cmd} --help is missing --gpu-budget-gb"


@pytest.mark.parametrize("cmd", _HEAVY_GPU_COMMANDS)
def test_command_help_lists_memory_planning_flags(cmd):
    txt = _help_text_for(cmd)
    # `--memory-diagnostics` was removed (always-on); replaced by
    # `--memory-profile` for the heavyweight JAX-profiler captures.
    expected = [
        "--low-memory-option",
        "--very-low-memory-option",
        "--adaptive-n-pcs",
        "--memory-profile",
        "--fail-on-memory-exceed",
    ]
    missing = [flag for flag in expected if flag not in txt]
    assert not missing, f"recovar {cmd} --help missing: {missing}"


@pytest.mark.parametrize("cmd", _HEAVY_GPU_COMMANDS)
def test_command_help_does_not_advertise_removed_aliases(cmd):
    """Catch alias resurrections."""
    txt = _help_text_for(cmd)
    forbidden = [
        "--gpu-memory",
        "--adaptive-memory",
        "--n-adaptive-pcs",
        "--hard-gpu-memory-limit",
        "--memory-diagnostics",  # removed: diagnostics are always-on
    ]
    present = [flag for flag in forbidden if flag in txt]
    assert not present, f"recovar {cmd} --help still advertises removed aliases: {present}"


def test_pipeline_parser_accepts_canonical_flags():
    from recovar.commands import pipeline as pipe

    parser = argparse.ArgumentParser()
    pipe.add_args(parser)

    args_gpu = parser.parse_args(["pp.mrcs", "-o", "/tmp", "--mask", "sphere", "--gpu-budget-gb", "12"])
    args_adaptive = parser.parse_args(["pp.mrcs", "-o", "/tmp", "--mask", "sphere", "--adaptive-n-pcs"])
    assert args_gpu.gpu_memory == 12.0
    assert args_adaptive.adaptive_memory is True


def test_positive_finite_gb_validator_rejects_bogus_values():
    """The shared argparse type rejects NaN / inf / 0 / negative."""

    from recovar.utils.parser_args import positive_finite_gb

    # Valid.
    assert positive_finite_gb("12.5") == 12.5
    assert positive_finite_gb("1") == 1.0

    # Invalid.
    import argparse

    for bad in ("NaN", "inf", "-inf", "-1", "0", "abc", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_finite_gb(bad)


def test_pipeline_parser_rejects_bogus_gpu_gb():
    """argparse must reject bogus --gpu-budget-gb values via the shared validator."""
    from recovar.commands import pipeline as pipe

    parser = argparse.ArgumentParser()
    pipe.add_args(parser)
    base = ["pp.mrcs", "-o", "/tmp", "--mask", "sphere"]
    for bad in ("NaN", "inf", "-1", "0"):
        with pytest.raises(SystemExit):
            parser.parse_args(base + ["--gpu-budget-gb", bad])


def test_pipeline_parser_rejects_removed_aliases():
    """argparse must reject the removed legacy names with SystemExit."""
    from recovar.commands import pipeline as pipe

    parser = argparse.ArgumentParser()
    pipe.add_args(parser)
    base = ["pp.mrcs", "-o", "/tmp", "--mask", "sphere"]
    cases = [
        ["--gpu-memory", "12"],
        ["--adaptive-memory"],
        ["--n-adaptive-pcs"],
        ["--hard-gpu-memory-limit"],
    ]
    for extra in cases:
        with pytest.raises(SystemExit):
            parser.parse_args(base + extra)


def test_run_test_dataset_always_splices_adaptive_n_pcs():
    """Default contract: an explicit ``--adaptive-n-pcs`` /
    ``--no-adaptive-n-pcs`` is ALWAYS spliced into the inner argv so
    the smoke test's choice is deterministic regardless of how the
    inner pipeline's own defaults evolve.

    Updated 2026-05-15: --adaptive-n-pcs is ON BY DEFAULT
    (BooleanOptionalAction). --full-memory-test now wins over
    --adaptive-n-pcs because the flag's whole point is to disable
    adaptive shrinking."""
    from recovar.commands import run_test_dataset as rtd

    parser = rtd._build_parser()

    # No flags at all — adaptive is now default-True, so we get --adaptive-n-pcs.
    args = parser.parse_args([])
    fwd = rtd._build_forward_argv(args)
    assert "--adaptive-n-pcs" in fwd
    assert "--no-adaptive-n-pcs" not in fwd

    # --gpu-budget-gb only — still spliced with adaptive ON.
    args = parser.parse_args(["--gpu-budget-gb", "32"])
    fwd = rtd._build_forward_argv(args)
    assert "--gpu-budget-gb" in fwd
    assert "32.0" in fwd
    assert "--adaptive-n-pcs" in fwd

    # --full-memory-test — opt out (splices --no-adaptive-n-pcs explicitly).
    args = parser.parse_args(["--gpu-budget-gb", "32", "--full-memory-test"])
    fwd = rtd._build_forward_argv(args)
    assert "--gpu-budget-gb" in fwd
    assert "--no-adaptive-n-pcs" in fwd
    assert "--adaptive-n-pcs" not in fwd  # only the --no- form is present

    # --no-adaptive-n-pcs explicit — opt out.
    args = parser.parse_args(["--no-adaptive-n-pcs"])
    fwd = rtd._build_forward_argv(args)
    assert "--no-adaptive-n-pcs" in fwd
    assert "--adaptive-n-pcs" not in fwd

    # --full-memory-test + --adaptive-n-pcs — --full-memory-test wins
    # (the user explicitly asked to disable shrinking).
    args = parser.parse_args(["--full-memory-test", "--adaptive-n-pcs"])
    fwd = rtd._build_forward_argv(args)
    assert "--no-adaptive-n-pcs" in fwd


def test_run_test_dataset_command_set_is_single_source_of_truth():
    """``_COMMANDS_WITH_MEMORY_ARGS`` must match the heavy-GPU list."""
    from recovar.commands import run_test_dataset as rtd

    expected = {
        "pipeline",
        "pipeline_with_outliers",
        "analyze",
        "compute_state",
        "compute_trajectory",
        "reconstruct_from_external_embedding",
        "junk_particle_detection",
        "outlier_detection",
    }
    assert set(rtd._COMMANDS_WITH_MEMORY_ARGS) == expected


# NOTE: the pre-import scan helpers (``_scan_for_flag_value`` /
# ``_scan_for_bool_flag``) used to live in ``recovar.command_line`` to
# convert --gpu-budget-gb into XLA_PYTHON_CLIENT_MEM_FRACTION before
# jax import. That auto-cap was reverted (orthogonal to --gpu-budget-gb;
# the user controls JAX-level allocation via the XLA env vars
# directly). The scan helpers were removed alongside.
