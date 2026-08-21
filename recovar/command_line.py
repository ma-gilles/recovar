"""Command-line entry point for the ``recovar`` CLI.

Dispatches ``recovar <command>`` to the matching module under
``recovar.commands``.  Each command module must define a ``main()``
function that uses ``argparse`` for its own argument parsing.

See ``[project.scripts]`` in ``pyproject.toml`` for the console-script
entry that invokes :func:`main_commands`.

One responsibility lives here around subcommand execution:

Hint-last error wrapping: when the subcommand fails, the captured
traceback is printed first, then the formatted ``ErrorHint``, so
the actionable advice stays at the tail of the log.

``--gpu-budget-gb`` is handled by argparse inside the subcommand —
it is a soft batch-size hint, NOT a JAX-level memory cap. Users on
shared / workstation GPUs who hit OOM with JAX's default
preallocation should ``export XLA_PYTHON_CLIENT_PREALLOCATE=false``;
that's a JAX deployment-mode question orthogonal to RECOVAR's
batch-size budget.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import traceback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------


def _print_available_commands(available_cmds, file=None):
    """Print the list of available commands."""
    file = file if file is not None else sys.stderr
    print("Available commands:", file=file)
    for cmd in available_cmds:
        print(f"  {cmd}", file=file)


def _run_with_error_hints(mod) -> None:
    """Run ``mod.main()`` and emit a hint-last error report on failure."""
    try:
        mod.main()
    except SystemExit:
        raise
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        try:
            from recovar.utils import error_hints

            ctx = error_hints.collect_context()
            hint = error_hints.classify_exception(exc, ctx)
            if hint is not None:
                sys.stderr.write("\n")
                sys.stderr.write(error_hints.format_error_hint(hint))
                sys.stderr.flush()
        except Exception as inner:
            logger.debug("Error-hint formatting failed: %s", inner)
        sys.exit(1)


def main_commands() -> None:
    """Primary entry point installed as ``recovar <cmd_module_name>``."""
    cmd_dir = os.path.join(os.path.dirname(__file__), "commands")

    available_cmds = sorted(
        os.path.splitext(f)[0] for f in os.listdir(cmd_dir) if f.endswith(".py") and f != "__init__.py"
    )

    if len(sys.argv) < 2:
        print("Usage: recovar <command>", file=sys.stderr)
        _print_available_commands(available_cmds)
        sys.exit(1)

    cmd_name = sys.argv[1]
    if cmd_name not in available_cmds:
        print(f"Command '{cmd_name}' not found.", file=sys.stderr)
        _print_available_commands(available_cmds)
        sys.exit(1)

    # Remove the subcommand from sys.argv so the subcommand's parser
    # doesn't see it.
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    module_name = f"recovar.commands.{cmd_name}"
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Error importing {module_name}: {e}", file=sys.stderr)
        sys.exit(1)

    if hasattr(mod, "main"):
        _run_with_error_hints(mod)
    else:
        print(f"Module {module_name} does not define a main() function.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main_commands()
