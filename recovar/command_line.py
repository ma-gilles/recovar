"""Command-line entry point for the ``recovar`` CLI.

Dispatches ``recovar <command>`` to the matching module under
``recovar.commands``.  Each command module must define a ``main()``
function that uses ``argparse`` for its own argument parsing.

See ``[project.scripts]`` in ``pyproject.toml`` for the console-script
entry that invokes :func:`main_commands`.

Two responsibilities live here that have to run BEFORE the subcommand
module is imported (i.e. before any ``import jax`` happens):

1. Memory bootstrap: a tiny tolerant scan of ``sys.argv`` for
   ``--hard-gpu-memory-limit`` + ``--gpu-gb`` so we can set
   ``XLA_PYTHON_CLIENT_MEM_FRACTION`` before jax initializes.
2. CUDA env-var typo detection: emit the warning eagerly so the user
   sees it before any error.

After the subcommand runs, any non-``SystemExit`` exception is captured
and the traceback is printed first, then the formatted error hint —
hint LAST is intentional so the user's eye lands on actionable advice,
not on the JAX traceback above it.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import traceback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap: hard GPU memory limit (must run before jax imports anywhere)
# ---------------------------------------------------------------------------


def _scan_for_flag_value(argv: list[str], names: tuple[str, ...]) -> str | None:
    """Return the value following any of ``names`` in argv, or None.

    Tolerates ``--flag=value`` and ``--flag value``. Does NOT consume
    argv — the real argparse layer in the subcommand still sees
    everything.
    """
    for i, tok in enumerate(argv):
        for name in names:
            if tok == name and i + 1 < len(argv):
                return argv[i + 1]
            if tok.startswith(name + "="):
                return tok.split("=", 1)[1]
    return None


def _scan_for_bool_flag(argv: list[str], names: tuple[str, ...]) -> bool:
    for tok in argv:
        for name in names:
            if tok == name:
                return True
            if tok.startswith(name + "="):
                return True
    return False


def _apply_hard_gpu_memory_limit(argv: list[str]) -> None:
    """If ``--hard-gpu-memory-limit`` is set, configure XLA before jax import.

    We compute the fraction from the user's ``--gpu-gb`` (or
    ``--gpu-memory``) divided by the physical GPU total, clamp into
    ``[0.05, 0.95]``, and export ``XLA_PYTHON_CLIENT_MEM_FRACTION``.

    No-ops gracefully when:
      - the flag is absent
      - no budget was given
      - NVML/nvidia-smi can't report the physical total
      - jax is somehow already imported (would be too late)
    """
    if not _scan_for_bool_flag(argv, ("--hard-gpu-memory-limit",)):
        return

    if "jax" in sys.modules:
        logger.warning(
            "Hard GPU memory limiting could not be applied because jax was "
            "already imported before command_line.py ran. RECOVAR will still "
            "use --gpu-gb as a soft planning budget."
        )
        return

    raw = _scan_for_flag_value(argv, ("--gpu-gb", "--gpu-memory"))
    if raw is None:
        logger.warning(
            "--hard-gpu-memory-limit was passed without --gpu-gb / --gpu-memory; no XLA memory cap will be applied."
        )
        return

    try:
        gpu_gb = float(raw)
    except ValueError:
        logger.warning("Could not parse --gpu-gb=%r as float; skipping hard limit.", raw)
        return

    safety_raw = _scan_for_flag_value(argv, ("--memory-safety-fraction",))
    safety = 1.0
    if safety_raw is not None:
        try:
            safety = float(safety_raw)
        except ValueError:
            logger.warning(
                "Could not parse --memory-safety-fraction=%r as float; using 1.0.",
                safety_raw,
            )

    physical_total: float | None = None
    try:
        from recovar.utils.gpu_preflight import get_physical_gpu_memory_info

        info = get_physical_gpu_memory_info(0)
        if info is not None:
            physical_total = info.total_gb
    except Exception as exc:
        logger.debug("Pre-jax physical-memory probe failed: %s", exc)

    if physical_total is None or physical_total <= 0:
        logger.warning(
            "Hard GPU memory limit requested but the physical GPU total is "
            "unknown (no pynvml/nvidia-smi). Falling back to a soft budget."
        )
        return

    fraction = (gpu_gb * safety) / physical_total
    fraction = max(0.05, min(0.95, fraction))

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{fraction:.4f}"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    logger.info(
        "Hard GPU memory limit applied: XLA_PYTHON_CLIENT_MEM_FRACTION=%.4f "
        "(--gpu-gb=%.1f, physical_total=%.1f GB, safety=%.2f, "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false)",
        fraction,
        gpu_gb,
        physical_total,
        safety,
    )


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------


def _print_available_commands(available_cmds, file=None):
    """Print the list of available commands."""
    file = file if file is not None else sys.stderr
    print("Available commands:", file=file)
    for cmd in available_cmds:
        print(f"  {cmd}", file=file)


def _eager_typo_warning() -> None:
    """Trigger the RECOVAR_CUDA_DISABLE typo detection eagerly.

    Calling ``custom_cuda_disabled_from_env`` here surfaces the warning
    (if any) at the top of stderr instead of waiting until something
    inside the pipeline reads the env var.
    """
    try:
        from recovar.utils.cuda_env import custom_cuda_disabled_from_env

        custom_cuda_disabled_from_env()
    except Exception as exc:
        logger.debug("Eager typo-warning probe failed: %s", exc)


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

    # Pre-import housekeeping that must run before jax_config.py loads:
    #   1. Apply hard GPU memory limit (sets XLA_PYTHON_CLIENT_MEM_FRACTION).
    #   2. Surface the RECOVAR_CUDA_DISABLE typo warning eagerly.
    _apply_hard_gpu_memory_limit(sys.argv[2:])
    _eager_typo_warning()

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
