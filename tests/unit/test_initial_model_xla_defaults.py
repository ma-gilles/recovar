"""InitialModel's CLI invocations reach XLA with the EM defaults.

``recovar/commands/initial_model.py`` sets ``RECOVAR_EM_XLA_DEFAULTS`` before its
own imports, but ``recovar initial_model`` (console script and GUI) and
``python -m recovar.commands.initial_model`` import the ``recovar`` package, and
with it ``recovar.jax_config``, before that module runs. The EM default
``--xla_gpu_autotune_level=0`` therefore never reached XLA for InitialModel,
and autotuning stayed on (with it XLA's fusion autotuner, whose Triton
reductions over-read their operands on Hopper; D2 10345 diagnosis). The package
now sets the marker itself when it is imported for the InitialModel CLI.

The end-to-end cases run the real invocations in child interpreters and read
``XLA_FLAGS`` at exit through a ``sitecustomize`` hook.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import recovar

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
MARKER = "RECOVAR_EM_XLA_DEFAULTS"
FLAG = "--xla_gpu_autotune_level=0"


@pytest.mark.parametrize(
    ("argv", "orig_argv"),
    [
        (["/env/bin/recovar", "initial_model", "--help"], ["/env/bin/python", "/env/bin/recovar", "initial_model"]),
        (["-m", "--help"], ["/env/bin/python", "-m", "recovar.commands.initial_model", "--help"]),
    ],
    ids=["console-script", "python-m"],
)
def test_the_initial_model_cli_sets_the_marker(argv, orig_argv):
    environ: dict[str, str] = {}
    assert recovar._configure_initial_model_xla_defaults(argv=argv, orig_argv=orig_argv, environ=environ) == "1"
    assert environ == {MARKER: "1"}


def test_other_commands_do_not_set_the_marker():
    environ: dict[str, str] = {}
    argv = ["/env/bin/recovar", "pipeline", "--help"]
    assert recovar._configure_initial_model_xla_defaults(argv=argv, orig_argv=argv, environ=environ) is None
    assert environ == {}


@pytest.mark.parametrize("explicit", ["0", ""])
def test_an_explicit_marker_wins(explicit):
    environ = {MARKER: explicit}
    argv = ["/env/bin/recovar", "initial_model"]
    assert recovar._configure_initial_model_xla_defaults(argv=argv, orig_argv=argv, environ=environ) == explicit


def _xla_flags_at_exit(cmd: list[str], tmp_path: Path, extra_env: dict[str, str] | None = None) -> str:
    hook = tmp_path / "sitecustomize.py"
    hook.write_text(
        "import atexit, os\n"
        "atexit.register(lambda: print('XLA_FLAGS_AT_EXIT=' + os.environ.get('XLA_FLAGS', ''), flush=True))\n"
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": f"{tmp_path}:{REPO}",
        "JAX_PLATFORMS": "cpu",
        "RECOVAR_DISABLE_CUDA": "1",
        "PYTHONNOUSERSITE": "1",
        "HOME": str(tmp_path),
        **(extra_env or {}),
    }
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("XLA_FLAGS_AT_EXIT=")]
    assert lines, proc.stdout[-2000:]
    return lines[-1].split("=", 1)[1]


def test_end_to_end_console_script_reaches_xla_with_the_em_default(tmp_path):
    code = (
        "import sys; sys.argv = ['recovar', 'initial_model', '--help'];"
        " from recovar.command_line import main_commands; main_commands()"
    )
    assert FLAG in _xla_flags_at_exit([sys.executable, "-c", code], tmp_path).split()


def test_end_to_end_python_m_reaches_xla_with_the_em_default(tmp_path):
    cmd = [sys.executable, "-m", "recovar.commands.initial_model", "--help"]
    assert FLAG in _xla_flags_at_exit(cmd, tmp_path).split()


def test_end_to_end_other_commands_keep_autotuning(tmp_path):
    code = "import sys; sys.argv = ['recovar', 'pipeline', '--help']; import recovar"
    assert FLAG not in _xla_flags_at_exit([sys.executable, "-c", code], tmp_path).split()
