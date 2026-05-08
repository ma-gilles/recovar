"""Command-line entry point for the ``recovar`` CLI.

Dispatches ``recovar <command>`` to the matching module under
``recovar.commands``.  Each command module must define a ``main()``
function that uses ``argparse`` for its own argument parsing.

See ``[project.scripts]`` in ``pyproject.toml`` for the console-script
entry that invokes :func:`main_commands`.

Two responsibilities live here that have to run BEFORE the subcommand
module is imported (i.e. before any ``import jax`` happens):

1. Memory bootstrap: when ``--gpu-gb N`` is on the command line, scan
   for it and set ``XLA_PYTHON_CLIENT_MEM_FRACTION = N / physical_total``
   before jax initializes — the actual fix for #135.
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


def _apply_gpu_memory_cap(argv: list[str]) -> None:
    """When ``--gpu-gb N`` is set, cap JAX's GPU allocation to N GB.

    This is the actual fix for issue #135: the original ``--gpu-gb``
    flag only affected RECOVAR's batch-size formulas, while JAX
    independently allocated up to ``XLA_PYTHON_CLIENT_MEM_FRACTION=.90``
    of physical VRAM (the default in ``recovar/jax_config.py``). On a
    48 GB workstation with no flag, JAX would lock up 43 GB before
    RECOVAR's planner ran. Now ``--gpu-gb 24`` translates to
    ``XLA_PYTHON_CLIENT_MEM_FRACTION = 24 / physical_total`` and JAX
    actually honors it.

    Done by a tolerant pre-import scan of argv (NOT argparse) so the
    real subcommand parser still sees ``--gpu-gb`` and feeds it to the
    planner for batch-size computation.

    No-ops gracefully when:
      - ``--gpu-gb`` is absent (default JAX behavior)
      - jax is somehow already imported (too late)
      - NVML/nvidia-smi can't report the physical total
    """
    raw = _scan_for_flag_value(argv, ("--gpu-gb",))
    if raw is None:
        return

    # Validate using the same rules as the argparse type so a malformed
    # ``--gpu-gb=NaN`` fails at the bootstrap layer too. The argparse
    # layer in the subcommand will then surface the same error to the
    # user once dispatch resumes.
    from recovar.utils.parser_args import positive_finite_gb

    try:
        gpu_gb = positive_finite_gb(raw)
    except Exception as exc:
        logger.warning(
            "Bootstrap could not validate --gpu-gb=%r (%s); deferring to argparse.",
            raw,
            exc,
        )
        return

    if "jax" in sys.modules:
        logger.warning(
            "GPU memory cap could not be applied because jax was already "
            "imported before command_line.py ran. RECOVAR will still use "
            "--gpu-gb as a soft planning budget for batch sizes, but JAX's "
            "memory allocation is no longer constrained."
        )
        os.environ["RECOVAR_GPU_CAP_NOT_APPLIED"] = "jax_already_imported"
        return

    physical_total: float | None = None
    n_visible_devices: int | None = None
    try:
        from recovar.utils.gpu_preflight import get_physical_gpu_memory_info

        info = get_physical_gpu_memory_info(0)
        if info is not None:
            physical_total = info.total_gb
        # Best-effort visible-device count for the multi-GPU log line.
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd:
            n_visible_devices = len([t for t in cvd.split(",") if t.strip()])
    except Exception as exc:
        logger.debug("Pre-jax physical-memory probe failed: %s", exc)

    if physical_total is None or physical_total <= 0:
        logger.warning(
            "--gpu-gb=%.1f was supplied but the physical GPU total could "
            "not be determined (no pynvml/nvidia-smi). JAX's allocation "
            "is NOT capped; --gpu-gb will only affect RECOVAR's batch-size "
            "formulas. Set RECOVAR_CUDA_VISIBLE_GPU_GB=<total_gb> to "
            "specify a value manually for the cap.",
            gpu_gb,
        )
        os.environ["RECOVAR_GPU_CAP_NOT_APPLIED"] = "no_nvml_or_nvidia_smi"
        return

    if gpu_gb > physical_total:
        logger.warning(
            "--gpu-gb=%.1f exceeds physical GPU total %.1f; capping at 0.95 of physical.",
            gpu_gb,
            physical_total,
        )

    fraction = gpu_gb / physical_total
    fraction = max(0.05, min(0.95, fraction))

    existing = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    if existing is not None:
        logger.warning(
            "XLA_PYTHON_CLIENT_MEM_FRACTION=%s is set in the environment; "
            "overriding with the value derived from --gpu-gb=%.1f -> %.4f. "
            "If you want your shell value to win, omit --gpu-gb.",
            existing,
            gpu_gb,
            fraction,
        )

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{fraction:.4f}"
    os.environ.pop("RECOVAR_GPU_CAP_NOT_APPLIED", None)

    if n_visible_devices and n_visible_devices > 1:
        logger.info(
            "GPU memory cap applied: XLA_PYTHON_CLIENT_MEM_FRACTION=%.4f "
            "(--gpu-gb=%.1f, physical_total=%.1f GB on GPU 0; "
            "%d visible devices — semantics: per visible device, no "
            "cross-device rebalancing).",
            fraction,
            gpu_gb,
            physical_total,
            n_visible_devices,
        )
    else:
        logger.info(
            "GPU memory cap applied: XLA_PYTHON_CLIENT_MEM_FRACTION=%.4f (--gpu-gb=%.1f, physical_total=%.1f GB)",
            fraction,
            gpu_gb,
            physical_total,
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
    #   1. When --gpu-gb is set, cap JAX's allocation to that budget
    #      via XLA_PYTHON_CLIENT_MEM_FRACTION (the actual fix for #135).
    #   2. Surface the RECOVAR_CUDA_DISABLE typo warning eagerly.
    _apply_gpu_memory_cap(sys.argv[2:])
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
