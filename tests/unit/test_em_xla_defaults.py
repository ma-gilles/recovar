"""P4-J: the EM-scoped XLA default (`--xla_gpu_autotune_level=0`).

Ticket: ``em_parity_tickets_20260918/P4J_compile_ahead_and_autotune_default.md``.

The policy is a pure function so it can be tested without importing jax, and
the two facts that make it "scoped to the EM entry points" are tested directly:

* a plain ``import recovar`` does not add the flag, because nothing sets the
  marker;
* the EM entry scripts (in relax) set the marker, and set it before the first
  import that reaches jax, because ``XLA_FLAGS`` is read at jax import time;
  relax's tests/unit/test_em_xla_defaults_entries.py checks them.

A user who has already chosen `xla_gpu_autotune_level` always wins, in either
direction, which is the same contract the neighbouring triton-GEMM default has.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from recovar.jax_config import em_xla_flag_additions

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
MARKER = "RECOVAR_EM_XLA_DEFAULTS"
FLAG = "--xla_gpu_autotune_level=0"


# ------------------------------------------------------------- the policy ---


@pytest.mark.parametrize("marker", ["1", "on", "true", "yes", "anything-else"])
def test_the_flag_is_added_when_an_em_entry_asks(marker):
    assert em_xla_flag_additions("", marker) == FLAG


@pytest.mark.parametrize("marker", ["", "0", "off", "false", "no", "  "])
def test_nothing_is_added_without_the_marker(marker):
    assert em_xla_flag_additions("", marker) == ""


def test_a_plain_import_recovar_does_not_set_the_marker():
    """The scoping claim: only an EM entry point opts in."""

    assert em_xla_flag_additions("", "") == ""


@pytest.mark.parametrize("existing", [
    "--xla_gpu_autotune_level=0",
    "--xla_gpu_autotune_level=4",
    "--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=2",
])
def test_the_users_own_choice_wins(existing):
    assert em_xla_flag_additions(existing, "1") == ""


def test_it_composes_with_the_triton_default():
    existing = "--xla_gpu_enable_triton_gemm=false"
    assert em_xla_flag_additions(existing, "1") == FLAG


def test_unrelated_flags_are_left_alone():
    existing = "--xla_force_host_platform_device_count=8"
    assert em_xla_flag_additions(existing, "1") == FLAG


# --------------------------------------------------------- end to end ---


def test_end_to_end_the_flag_reaches_xla_flags_only_for_an_em_entry():
    """Run two child interpreters: one plain, one with the marker set."""

    script = (
        "import os, sys; sys.path.insert(0, %r);\n"
        "import recovar.jax_config as jc;\n"
        "print(os.environ.get('XLA_FLAGS',''))\n" % str(REPO)
    )
    env_common = {
        "PATH": "/usr/bin:/bin",
        "JAX_PLATFORMS": "cpu",
        "RECOVAR_DISABLE_CUDA": "1",
        "PYTHONNOUSERSITE": "1",
        "HOME": "/tmp",
    }
    plain = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, env=dict(env_common), timeout=600)
    assert plain.returncode == 0, plain.stderr[-2000:]
    assert FLAG not in plain.stdout, f"a plain import added the flag: {plain.stdout!r}"

    em = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                        env=dict(env_common, **{MARKER: "1"}), timeout=600)
    assert em.returncode == 0, em.stderr[-2000:]
    assert FLAG in em.stdout, f"the EM entry did not add the flag: {em.stdout!r}"
