"""Command-line entry point for the ``recovar`` CLI.

Dispatches ``recovar <command>`` to the matching module under
``recovar.commands``.  Each command module must define a ``main()``
function that uses ``argparse`` for its own argument parsing.

See ``[project.scripts]`` in ``pyproject.toml`` for the console-script
entry that invokes :func:`main_commands`.
"""

import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _print_available_commands(available_cmds, file=None):
    """Print the list of available commands."""
    file = file if file is not None else sys.stderr
    print("Available commands:", file=file)
    for cmd in available_cmds:
        print(f"  {cmd}", file=file)


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
        mod.main()
    else:
        print(f"Module {module_name} does not define a main() function.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main_commands()
